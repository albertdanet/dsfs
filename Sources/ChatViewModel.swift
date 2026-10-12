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
    @Published var status = "Загрузка модели…"
    @Published var systemPrompt: String {
        didSet { UserDefaults.standard.set(systemPrompt, forKey: "systemPrompt") }
    }

    private let engine = LlamaEngine.shared

    private var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
    private var chatFile: URL { documents.appendingPathComponent("chat.json") }
    private var userModelURL: URL { documents.appendingPathComponent("model.gguf") }

    init() {
        systemPrompt = UserDefaults.standard.string(forKey: "systemPrompt") ?? Self.defaultSystemPrompt
        loadChat()
        loadModelIfAvailable()
    }

    // MARK: - Модель

    private func locateModel() -> String? {
        // Импортированная пользователем модель важнее встроенной.
        if FileManager.default.fileExists(atPath: userModelURL.path) { return userModelURL.path }
        return Bundle.main.path(forResource: "model", ofType: "gguf")
    }

    func loadModelIfAvailable() {
        guard let path = locateModel() else {
            modelReady = false
            status = "Модель не найдена. Откройте настройки и импортируйте файл .gguf."
            return
        }
        modelReady = false
        status = "Загрузка модели…"
        engine.loadModel(path: path) { [weak self] result in
            guard let self = self else { return }
            switch result {
            case .success:
                self.modelReady = true
                self.status = "Готово · работает офлайн"
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

        var turns = [ChatTurn(role: "system", content: systemPrompt)]
        for m in messages.dropLast() {
            turns.append(ChatTurn(role: m.role, content: m.text))
        }

        isGenerating = true
        engine.generate(turns: turns, maxNewTokens: 512, onToken: { [weak self] piece in
            guard let self = self, replyIndex < self.messages.count else { return }
            self.messages[replyIndex].text += piece
        }, onFinish: { [weak self] error in
            guard let self = self else { return }
            self.isGenerating = false
            if error != nil, replyIndex < self.messages.count, self.messages[replyIndex].text.isEmpty {
                self.messages[replyIndex].text = "⚠️ Не удалось сгенерировать ответ."
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
