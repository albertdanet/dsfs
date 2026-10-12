import Foundation

struct ChatTurn {
    let role: String      // "system" | "user" | "assistant"
    let content: String
}

enum EngineError: Error {
    case loadFailed
    case noModel
    case contextFailed
    case decodeFailed
}

/// Тонкая обёртка над llama.cpp (C API, тег b5046). Только CPU, без Metal — так работает на iOS 15.
final class LlamaEngine {
    static let shared = LlamaEngine()

    let contextSize = 2048
    private let batchSize = 512

    private var model: OpaquePointer?
    private var backendReady = false
    private let queue = DispatchQueue(label: "llama.engine", qos: .userInitiated)
    private let lock = NSLock()
    private var cancelled = false

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

    func loadModel(path: String, completion: @escaping (Result<Void, Error>) -> Void) {
        queue.async {
            if !self.backendReady {
                llama_backend_init()
                self.backendReady = true
            }
            if let old = self.model {
                llama_model_free(old)
                self.model = nil
            }
            var params = llama_model_default_params()
            params.n_gpu_layers = 0
            params.use_mmap = true
            guard let loaded = llama_model_load_from_file(path, params) else {
                DispatchQueue.main.async { completion(.failure(EngineError.loadFailed)) }
                return
            }
            self.model = loaded
            DispatchQueue.main.async { completion(.success(())) }
        }
    }

    // MARK: - Генерация

    func generate(turns: [ChatTurn],
                  maxNewTokens: Int = 512,
                  onToken: @escaping (String) -> Void,
                  onFinish: @escaping (Error?) -> Void) {
        resetCancel()
        queue.async {
            let finish: (Error?) -> Void = { err in
                DispatchQueue.main.async { onFinish(err) }
            }

            guard let model = self.model, let vocab = llama_model_get_vocab(model) else {
                finish(EngineError.noModel)
                return
            }

            // Собираем промпт (ChatML — формат Qwen2.5) и режем старые сообщения, если не влезает.
            var work = turns
            var tokens = self.tokenize(vocab, self.buildPrompt(work))
            let budget = self.contextSize - maxNewTokens - 8
            while tokens.count > budget {
                guard work.count > 2,
                      let idx = work.firstIndex(where: { $0.role != "system" }) else { break }
                work.remove(at: idx)
                tokens = self.tokenize(vocab, self.buildPrompt(work))
            }
            if tokens.count > budget {
                tokens = Array(tokens.suffix(budget))
            }
            if tokens.isEmpty {
                finish(EngineError.decodeFailed)
                return
            }

            // Контекст создаём на каждый ответ — так не нужен API очистки KV-кэша.
            var cparams = llama_context_default_params()
            cparams.n_ctx = UInt32(self.contextSize)
            cparams.n_batch = UInt32(self.batchSize)
            let threads = Int32(min(4, max(2, ProcessInfo.processInfo.activeProcessorCount - 2)))
            cparams.n_threads = threads
            cparams.n_threads_batch = threads

            guard let ctx = llama_init_from_model(model, cparams) else {
                finish(EngineError.contextFailed)
                return
            }
            defer { llama_free(ctx) }

            // Сэмплер
            let sparams = llama_sampler_chain_default_params()
            guard let sampler = llama_sampler_chain_init(sparams) else {
                finish(EngineError.contextFailed)
                return
            }
            defer { llama_sampler_free(sampler) }
            llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, 1.1, 0.0, 0.0))
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40))
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1))
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7))
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(UInt32.random(in: 0..<UInt32.max)))

            // Prefill кусками по batchSize
            var pos = 0
            while pos < tokens.count {
                if self.isCancelled() { finish(nil); return }
                let end = min(pos + self.batchSize, tokens.count)
                var chunk = Array(tokens[pos..<end])
                let rc = chunk.withUnsafeMutableBufferPointer { buf -> Int32 in
                    let batch = llama_batch_get_one(buf.baseAddress, Int32(buf.count))
                    return llama_decode(ctx, batch)
                }
                if rc != 0 { finish(EngineError.decodeFailed); return }
                pos = end
            }

            // Генерация
            var pending = [UInt8]()
            var produced = 0
            while produced < maxNewTokens {
                if self.isCancelled() { break }
                var next = llama_sampler_sample(sampler, ctx, -1)
                if llama_vocab_is_eog(vocab, next) { break }

                pending.append(contentsOf: self.pieceBytes(vocab, next))
                if let s = String(bytes: pending, encoding: .utf8) {
                    pending.removeAll(keepingCapacity: true)
                    if s.contains("<|im_end|>") { break }
                    DispatchQueue.main.async { onToken(s) }
                } else if pending.count >= 8 {
                    // Невалидная последовательность — не зависаем, выводим как есть.
                    let s = String(decoding: pending, as: UTF8.self)
                    pending.removeAll(keepingCapacity: true)
                    DispatchQueue.main.async { onToken(s) }
                }

                let rc = withUnsafeMutablePointer(to: &next) { p -> Int32 in
                    let batch = llama_batch_get_one(p, 1)
                    return llama_decode(ctx, batch)
                }
                if rc != 0 { break }
                produced += 1
            }
            if !pending.isEmpty {
                let s = String(decoding: pending, as: UTF8.self)
                DispatchQueue.main.async { onToken(s) }
            }
            finish(nil)
        }
    }

    // MARK: - Помощники

    private func buildPrompt(_ turns: [ChatTurn]) -> String {
        var s = ""
        for t in turns {
            s += "<|im_start|>\(t.role)\n\(t.content)<|im_end|>\n"
        }
        s += "<|im_start|>assistant\n"
        return s
    }

    private func tokenize(_ vocab: OpaquePointer, _ text: String) -> [llama_token] {
        let utf8Count = text.utf8.count
        var tokens = [llama_token](repeating: 0, count: utf8Count + 16)
        var n = llama_tokenize(vocab, text, Int32(utf8Count), &tokens, Int32(tokens.count), false, true)
        if n < 0 {
            tokens = [llama_token](repeating: 0, count: Int(-n))
            n = llama_tokenize(vocab, text, Int32(utf8Count), &tokens, Int32(tokens.count), false, true)
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
