import SwiftUI

struct ChatMessage: Identifiable, Codable, Equatable {
    var id = UUID()
    var role: String   // "user" | "assistant"
    var text: String
}

final class ChatViewModel: ObservableObject {
    static let defaultSystemPrompt =
        "Ты — полезный ИИ-ассистент. Отвечай на языке пользователя, понятно и по делу."

    @Published var messages: [ChatMessage] = []
    @Published var input = ""
    @Published var isGenerating = false
    @Published var modelReady = false
    @Published var isReasoning = false
    @Published var status = "Загрузка модели…"
    @Published var systemPrompt: String {
        didSet { UserDefaults.standard.set(systemPrompt, forKey: "systemPrompt") }
    }
    /// 0 = автоматически (число быстрых ядер процессора)
    @Published var threads: Int {
        didSet { UserDefaults.standard.set(threads, forKey: "threads") }
    }

    private let engine = LlamaEngine.shared

    private var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
    private var chatFile: URL { documents.appendingPathComponent("chat.json") }
    private var userModelURL: URL { documents.appendingPathComponent("model.gguf") }

    init() {
        systemPrompt = UserDefaults.standard.string(forKey: "systemPrompt") ?? Self.defaultSystemPrompt
        threads = UserDefaults.standard.integer(forKey: "threads")
        loadChat()
        loadModelIfAvailable()
    }

    // MARK: - Модель

    func loadModelIfAvailable() {
        // Импортированная пользователем модель важнее встроенной.
        let hasUserModel = FileManager.default.fileExists(atPath: userModelURL.path)
        let bundled = Bundle.main.path(forResource: "model", ofType: "gguf")
        let path = hasUserModel ? userModelURL.path : bundled

        guard let modelPath = path else {
            modelReady = false
            status = "Модель не найдена. Откройте настройки и импортируйте файл .gguf."
            return
        }
        // Сборка с DeepSeek кладёт рядом файл-метку reasoning.txt
        let marker = Bundle.main.path(forResource: "reasoning", ofType: "txt") != nil
        let forceReasoning = !hasUserModel && marker

        modelReady = false
        status = "Загрузка модели…"
        engine.loadModel(path: modelPath, forceReasoning: forceReasoning) { [weak self] result in
            guard let self = self else { return }
            switch result {
            case .success:
                self.isReasoning = self.engine.isReasoning
                self.modelReady = true
                self.status = self.isReasoning
                    ? "Готово · режим рассуждений · офлайн"
                    : "Готово · работает офлайн"
            case .failure:
                self.status = "Не удалось загрузить модель (файл повреждён или не подходит)."
            }
        }
    }

    func importModel(from url: URL) {
        status = "Копирование модели…"
        modelReady = false
        let dest = userModelURL
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let access = url.startAccessingSecurityScopedResource()
            defer { if access { url.stopAccessingSecurityScopedResource() } }
            do {
                if FileManager.default.fileExists(atPath: dest.path) {
                    try FileManager.default.removeItem(at: dest)
                }
                try FileManager.default.copyItem(at: url, to: dest)
                DispatchQueue.main.async { self?.loadModelIfAvailable() }
            } catch {
                DispatchQueue.main.async {
                    self?.status = "Не удалось скопировать файл: \(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - Чат

    func send() {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, modelReady, !isGenerating else { return }
        input = ""

        messages.append(ChatMessage(role: "user", text: text))
        messages.append(ChatMessage(role: "assistant", text: ""))
        let replyIndex = messages.count - 1

        var turns: [ChatTurn] = []
        if !systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            turns.append(ChatTurn(role: "system", content: systemPrompt))
        }
        for m in messages.dropLast() {
            if m.role == "assistant" {
                // В историю идёт только ответ, без рассуждений.
                let answer = ThinkParser.parse(m.text, reasoning: isReasoning).answer
                if answer.isEmpty { continue }
                turns.append(ChatTurn(role: "assistant", content: answer))
            } else {
                turns.append(ChatTurn(role: "user", content: m.text))
            }
        }

        let nThreads = threads == 0 ? LlamaEngine.autoThreads() : threads
        isGenerating = true
        engine.generate(turns: turns, threads: nThreads, onToken: { [weak self] piece in
            guard let self = self, replyIndex < self.messages.count else { return }
            self.messages[replyIndex].text += piece
        }, onFinish: { [weak self] error, stats in
            guard let self = self else { return }
            self.isGenerating = false
            if error != nil, replyIndex < self.messages.count, self.messages[replyIndex].text.isEmpty {
                self.messages[replyIndex].text = "⚠️ Не удалось сгенерировать ответ."
            }
            if let s = stats, s.tokens > 0 {
                self.status = String(format: "Готово · %.1f ток/с · офлайн", s.tokensPerSecond)
            }
            self.saveChat()
        })
    }

    func stop() {
        engine.cancel()
    }

    func newChat() {
        if isGenerating { engine.cancel() }
        messages = []
        saveChat()
    }

    // MARK: - Сохранение

    private func saveChat() {
        if let data = try? JSONEncoder().encode(messages) {
            try? data.write(to: chatFile)
        }
    }

    private func loadChat() {
        guard let data = try? Data(contentsOf: chatFile),
              let saved = try? JSONDecoder().decode([ChatMessage].self, from: data) else { return }
        messages = saved
    }
}
