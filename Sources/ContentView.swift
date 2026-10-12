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
                                MessageBubble(message: message)
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

    var body: some View {
        let isUser = message.role == "user"
        HStack {
            if isUser { Spacer(minLength: 40) }
            Text(message.text.isEmpty ? "…" : message.text)
                .textSelection(.enabled)
                .padding(10)
                .background(isUser ? Color.accentColor : Color(.secondarySystemBackground))
                .foregroundColor(isUser ? .white : .primary)
                .cornerRadius(14)
            if !isUser { Spacer(minLength: 40) }
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
                Section(header: Text("Системный промпт")) {
                    TextEditor(text: $vm.systemPrompt)
                        .frame(minHeight: 120)
                    Button("Сбросить по умолчанию") {
                        vm.systemPrompt = ChatViewModel.defaultSystemPrompt
                    }
                }
                Section(header: Text("Модель"),
                        footer: Text("Можно выбрать любой файл .gguf в формате ChatML (например, Qwen2.5 Instruct, Q4_0). Он копируется в приложение и работает без интернета.")) {
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
