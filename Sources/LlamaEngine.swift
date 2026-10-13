import Foundation

struct ChatTurn {
    let role: String      // "system" | "user" | "assistant"
    let content: String
}

struct GenStats {
    let tokens: Int
    let seconds: Double
    var tokensPerSecond: Double { seconds > 0 ? Double(tokens) / seconds : 0 }
}

enum EngineError: Error {
    case loadFailed
    case noModel
    case contextFailed
    case decodeFailed
}

/// Обёртка над llama.cpp (C API, тег b5046). Только CPU, без Metal — так работает на iOS 15.
/// Контекст и KV-кэш живут между ответами: на каждом сообщении досчитываются только новые токены,
/// а не вся история заново. Это главное ускорение в длинном диалоге.
final class LlamaEngine {
    static let shared = LlamaEngine()

    private(set) var isReasoning = false
    private(set) var contextSize = 2048

    private let batchSize = 256
    private var model: OpaquePointer?
    private var ctx: OpaquePointer?
    private var cached: [llama_token] = []   // токены, чей KV уже лежит в контексте
    private var backendReady = false
    private let queue = DispatchQueue(label: "llama.engine", qos: .userInitiated)
    private let lock = NSLock()
    private var cancelled = false

    // MARK: - Потоки

    /// Число «быстрых» ядер. На iPhone это обычно 2: медленные ядра только тормозят вычисления.
    static func autoThreads() -> Int {
        var n: Int32 = 0
        var size = MemoryLayout<Int32>.size
        if sysctlbyname("hw.perflevel0.physicalcpu", &n, &size, nil, 0) == 0, n > 0 {
            return Int(n)
        }
        return max(1, min(2, ProcessInfo.processInfo.activeProcessorCount))
    }

    // MARK: - Отмена

    func cancel() {
        lock.lock(); cancelled = true; lock.unlock()
    }

    private func resetCancel() {
        lock.lock(); cancelled = false; lock.unlock()
    }

    private func isCancelled() -> Bool {
        lock.lock(); defer { lock.unlock() }
        return cancelled
    }

    // MARK: - Загрузка модели

    func loadModel(path: String,
                   forceReasoning: Bool,
                   completion: @escaping (Result<Void, Error>) -> Void) {
        queue.async {
            if !self.backendReady {
                llama_backend_init()
                self.backendReady = true
            }
            if let c = self.ctx { llama_free(c); self.ctx = nil }
            if let m = self.model { llama_model_free(m); self.model = nil }
            self.cached = []

            var mparams = llama_model_default_params()
            mparams.n_gpu_layers = 0
            mparams.use_mmap = true
            guard let loaded = llama_model_load_from_file(path, mparams) else {
                DispatchQueue.main.async { completion(.failure(EngineError.loadFailed)) }
                return
            }

            // Модели типа DeepSeek-R1 сначала «думают» в <think>…</think>.
            var nameBuf = [CChar](repeating: 0, count: 256)
            let len = llama_model_meta_val_str(loaded, "general.name", &nameBuf, nameBuf.count)
            let name = len > 0
                ? nameBuf.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }.lowercased()
                : ""
            let looksLikeR1 = name.contains("r1") && (name.contains("deepseek") || name.contains("distill"))
            self.isReasoning = forceReasoning || looksLikeR1
            self.contextSize = self.isReasoning ? 4096 : 2048

            var cparams = llama_context_default_params()
            cparams.n_ctx = UInt32(self.contextSize)
            cparams.n_batch = UInt32(self.batchSize)
            cparams.n_ubatch = UInt32(self.batchSize)
            let t = Int32(LlamaEngine.autoThreads())
            cparams.n_threads = t
            cparams.n_threads_batch = t
            guard let c = llama_init_from_model(loaded, cparams) else {
                llama_model_free(loaded)
                DispatchQueue.main.async { completion(.failure(EngineError.contextFailed)) }
                return
            }
            self.model = loaded
            self.ctx = c
            DispatchQueue.main.async { completion(.success(())) }
        }
    }

    // MARK: - Генерация

    func generate(turns: [ChatTurn],
                  threads: Int,
                  onToken: @escaping (String) -> Void,
                  onFinish: @escaping (Error?, GenStats?) -> Void) {
        resetCancel()
        queue.async {
            let finish: (Error?, GenStats?) -> Void = { err, stats in
                DispatchQueue.main.async { onFinish(err, stats) }
            }

            guard let model = self.model, let ctx = self.ctx,
                  let vocab = llama_model_get_vocab(model) else {
                finish(EngineError.noModel, nil)
                return
            }

            let maxNew = self.isReasoning ? 1536 : 512
            let budget = self.contextSize - maxNew - 8

            // Если история не влезает, выбрасываем самые старые сообщения.
            var work = turns
            var tokens = self.tokenize(vocab, self.buildPrompt(model, work))
            while tokens.count > budget {
                guard work.count > 2,
                      let idx = work.firstIndex(where: { $0.role != "system" }) else { break }
                work.remove(at: idx)
                tokens = self.tokenize(vocab, self.buildPrompt(model, work))
            }
            if tokens.count > budget {
                tokens = Array(tokens.suffix(budget))
            }
            guard !tokens.isEmpty else {
                finish(EngineError.decodeFailed, nil)
                return
            }

            // Переиспользуем KV-кэш: считаем заново только то, что отличается от прошлого раза.
            var common = self.commonPrefixLength(self.cached, tokens)
            if common >= tokens.count { common = tokens.count - 1 }
            if common < self.cached.count {
                if !llama_kv_self_seq_rm(ctx, 0, Int32(common), -1) {
                    llama_kv_self_clear(ctx)
                    common = 0
                }
                self.cached = Array(self.cached.prefix(common))
            }

            let nThreads = Int32(max(1, threads))
            llama_set_n_threads(ctx, nThreads, nThreads)

            // Сэмплер
            guard let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params()) else {
                finish(EngineError.contextFailed, nil)
                return
            }
            defer { llama_sampler_free(sampler) }
            if self.isReasoning {
                llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, 1.05, 0.0, 0.0))
                llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40))
                llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95, 1))
                llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.6))
            } else {
                llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, 1.1, 0.0, 0.0))
                llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40))
                llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1))
                llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7))
            }
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(UInt32.random(in: 0..<UInt32.max)))

            // Prefill только новых токенов, кусками по batchSize
            var pos = common
            while pos < tokens.count {
                if self.isCancelled() { finish(nil, nil); return }
                let end = min(pos + self.batchSize, tokens.count)
                var chunk = Array(tokens[pos..<end])
                let rc = chunk.withUnsafeMutableBufferPointer { buf -> Int32 in
                    llama_decode(ctx, llama_batch_get_one(buf.baseAddress, Int32(buf.count)))
                }
                if rc != 0 {
                    llama_kv_self_clear(ctx)
                    self.cached = []
                    finish(EngineError.decodeFailed, nil)
                    return
                }
                self.cached.append(contentsOf: chunk)
                pos = end
            }

            // Генерация. Текст отправляем в интерфейс пачками (~12 раз в секунду):
            // частые обновления SwiftUI отнимают процессор у самой модели.
            var pending = [UInt8]()
            var outBuf = ""
            var lastFlush = Date()
            var produced = 0
            let t0 = Date()

            func flush() {
                guard !outBuf.isEmpty else { return }
                let s = outBuf
                outBuf = ""
                lastFlush = Date()
                DispatchQueue.main.async { onToken(s) }
            }

            while produced < maxNew {
                if self.isCancelled() { break }
                if self.cached.count >= self.contextSize - 1 { break }

                var next = llama_sampler_sample(sampler, ctx, -1)
                if llama_vocab_is_eog(vocab, next) { break }

                pending.append(contentsOf: self.pieceBytes(vocab, next))
                if let s = String(bytes: pending, encoding: .utf8) {
                    pending.removeAll(keepingCapacity: true)
                    outBuf += s
                } else if pending.count >= 8 {
                    outBuf += String(decoding: pending, as: UTF8.self)
                    pending.removeAll(keepingCapacity: true)
                }
                if Date().timeIntervalSince(lastFlush) > 0.08 { flush() }

                let rc = withUnsafeMutablePointer(to: &next) { p -> Int32 in
                    llama_decode(ctx, llama_batch_get_one(p, 1))
                }
                if rc != 0 {
                    llama_kv_self_clear(ctx)
                    self.cached = []
                    break
                }
                self.cached.append(next)
                produced += 1
            }

            if !pending.isEmpty {
                outBuf += String(decoding: pending, as: UTF8.self)
            }
            flush()
            let seconds = Date().timeIntervalSince(t0)
            finish(nil, GenStats(tokens: produced, seconds: seconds))
        }
    }

    // MARK: - Промпт

    /// Шаблон берём из самой модели (Qwen, DeepSeek и т.д.); если его нет, используем ChatML.
    private func buildPrompt(_ model: OpaquePointer, _ turns: [ChatTurn]) -> String {
        var text = applyTemplate(model, turns) ?? chatML(turns)
        if isReasoning && !text.hasSuffix("<think>\n") {
            text += "<think>\n"   // так рекомендуют авторы DeepSeek-R1: модель сразу начинает рассуждать
        }
        return text
    }

    private func applyTemplate(_ model: OpaquePointer, _ turns: [ChatTurn]) -> String? {
        let tmpl = llama_model_chat_template(model, nil)
        var cStrings: [UnsafeMutablePointer<CChar>] = []
        var msgs: [llama_chat_message] = []
        for t in turns {
            guard let r = strdup(t.role), let c = strdup(t.content) else { continue }
            cStrings.append(r)
            cStrings.append(c)
            msgs.append(llama_chat_message(role: UnsafePointer(r), content: UnsafePointer(c)))
        }
        defer { cStrings.forEach { free($0) } }
        guard !msgs.isEmpty else { return nil }

        var cap = max(1024, turns.reduce(0) { $0 + $1.content.utf8.count + 64 } * 2)
        var buf = [CChar](repeating: 0, count: cap)
        var n = llama_chat_apply_template(tmpl, msgs, msgs.count, true, &buf, Int32(cap))
        if n > Int32(cap) {
            cap = Int(n) + 1
            buf = [CChar](repeating: 0, count: cap)
            n = llama_chat_apply_template(tmpl, msgs, msgs.count, true, &buf, Int32(cap))
        }
        guard n > 0 else { return nil }
        let bytes = buf[0..<Int(n)].map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }

    private func chatML(_ turns: [ChatTurn]) -> String {
        var s = ""
        for t in turns {
            s += "<|im_start|>\(t.role)\n\(t.content)<|im_end|>\n"
        }
        s += "<|im_start|>assistant\n"
        return s
    }

    // MARK: - Помощники

    private func commonPrefixLength(_ a: [llama_token], _ b: [llama_token]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n && a[i] == b[i] { i += 1 }
        return i
    }

    private func tokenize(_ vocab: OpaquePointer, _ text: String) -> [llama_token] {
        let utf8Count = text.utf8.count
        var tokens = [llama_token](repeating: 0, count: utf8Count + 16)
        var n = llama_tokenize(vocab, text, Int32(utf8Count), &tokens, Int32(tokens.count), true, true)
        if n < 0 {
            tokens = [llama_token](repeating: 0, count: Int(-n))
            n = llama_tokenize(vocab, text, Int32(utf8Count), &tokens, Int32(tokens.count), true, true)
        }
        return n > 0 ? Array(tokens.prefix(Int(n))) : []
    }

    private func pieceBytes(_ vocab: OpaquePointer, _ token: llama_token) -> [UInt8] {
        var buf = [CChar](repeating: 0, count: 64)
        var n = llama_token_to_piece(vocab, token, &buf, Int32(buf.count), 0, false)
        if n < 0 {
            buf = [CChar](repeating: 0, count: Int(-n))
            n = llama_token_to_piece(vocab, token, &buf, Int32(buf.count), 0, false)
        }
        guard n > 0 else { return [] }
        return buf[0..<Int(n)].map { UInt8(bitPattern: $0) }
    }
}
