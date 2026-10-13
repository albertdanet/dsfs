import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var vm = ChatViewModel()
    @State private var showSettings = false
    @State private var showImporter = false

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                Text(vm.status)
                    .font(.footnote)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 4)

                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 10) {
                            ForEach(vm.messages) { message in
                                MessageBubble(message: message, reasoning: vm.isReasoning)
                                    .id(message.id)
                            }
                            Color.clear.frame(height: 1).id("bottom")
                        }
                        .padding(.horizontal, 12)
                        .padding(.top, 8)
                    }
                    .onChange(of: vm.messages) { _ in
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                    .onTapGesture { hideKeyboard() }
                }

                Divider()
                HStack(spacing: 8) {
                    TextField("Сообщение…", text: $vm.input)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .submitLabel(.send)
                        .onSubmit { vm.send() }

                    if vm.isGenerating {
                        Button(action: vm.stop) {
                            Image(systemName: "stop.circle.fill").font(.system(size: 30))
                        }
                    } else {
                        Button(action: vm.send) {
                            Image(systemName: "arrow.up.circle.fill").font(.system(size: 30))
                        }
                        .disabled(!vm.modelReady || vm.input.trimmingCharacters(in: .whitespaces).isEmpty)
                    }
                }
                .padding(10)
            }
            .navigationTitle("Offline AI")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button { vm.newChat() } label: { Image(systemName: "square.and.pencil") }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button { showSettings = true } label: { Image(systemName: "gearshape") }
                }
            }
        }
        .navigationViewStyle(.stack)
        .sheet(isPresented: $showSettings) {
            SettingsView(vm: vm, showImporter: $showImporter, isPresented: $showSettings)
        }
        .fileImporter(isPresented: $showImporter,
                      allowedContentTypes: [.data],
                      allowsMultipleSelection: false) { result in
            if case .success(let urls) = result, let url = urls.first {
                vm.importModel(from: url)
            }
        }
    }

    private func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                        to: nil, from: nil, for: nil)
    }
}

struct MessageBubble: View {
    let message: ChatMessage
    let reasoning: Bool

    var body: some View {
        let isUser = message.role == "user"
        HStack {
            if isUser { Spacer(minLength: 40) }
            content(isUser: isUser)
                .padding(10)
                .background(isUser ? Color.accentColor : Color(.secondarySystemBackground))
                .foregroundColor(isUser ? .white : .primary)
                .cornerRadius(14)
            if !isUser { Spacer(minLength: 40) }
        }
    }

    @ViewBuilder
    private func content(isUser: Bool) -> some View {
        if isUser {
            Text(message.text).textSelection(.enabled)
        } else {
            let p = ThinkParser.parse(message.text, reasoning: reasoning)
            VStack(alignment: .leading, spacing: 6) {
                if !p.thinking.isEmpty {
                    if p.thinkingDone {
                        DisclosureGroup("💭 Размышления") {
                            Text(p.thinking)
                                .font(.footnote)
                                .foregroundColor(.secondary)
                        }
                    } else {
                        Text("💭 Думаю…")
                            .font(.footnote)
                            .foregroundColor(.secondary)
                        Text(p.thinking)
                            .font(.footnote)
                            .foregroundColor(.secondary)
                    }
                }
                if !p.answer.isEmpty {
                    Text(p.answer).textSelection(.enabled)
                } else if p.thinking.isEmpty {
                    Text("…")
                }
            }
        }
    }
}

struct SettingsView: View {
    @ObservedObject var vm: ChatViewModel
    @Binding var showImporter: Bool
    @Binding var isPresented: Bool

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Скорость"),
                        footer: Text("Авто = число быстрых ядер процессора (на iPhone обычно 2). Если ответы медленные, попробуй 2 или 3: больше потоков не значит быстрее, медленные ядра только мешают. Применяется со следующего ответа.")) {
                    Stepper(value: $vm.threads, in: 0...6) {
                        Text(vm.threads == 0 ? "Потоки: авто" : "Потоки: \(vm.threads)")
                    }
                }
                Section(header: Text("Системный промпт"),
                        footer: Text("Для модели DeepSeek его лучше оставить пустым.")) {
                    TextEditor(text: $vm.systemPrompt)
                        .frame(minHeight: 120)
                    Button("Сбросить по умолчанию") {
                        vm.systemPrompt = ChatViewModel.defaultSystemPrompt
                    }
                }
                Section(header: Text("Модель"),
                        footer: Text("Можно выбрать любой файл .gguf (Qwen2.5 Instruct, DeepSeek-R1-Distill и т.п., лучше квантизация Q4_0). Он копируется в приложение и работает без интернета.")) {
                    Button("Импортировать .gguf из «Файлов»") {
                        isPresented = false
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                            showImporter = true
                        }
                    }
                    Text(vm.status).font(.footnote).foregroundColor(.secondary)
                }
            }
            .navigationTitle("Настройки")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Готово") { isPresented = false }
                }
            }
        }
        .navigationViewStyle(.stack)
    }
}
