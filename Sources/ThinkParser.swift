import Foundation

struct ParsedReply {
    var thinking: String
    var answer: String
    var thinkingDone: Bool
}

/// Разбирает ответ модели вроде DeepSeek-R1: «<think>рассуждения</think> ответ».
/// Если приложение само добавило «<think>» в промпт, открывающего тега в ответе нет —
/// тогда флаг `reasoning` говорит, что весь текст до «</think>» это рассуждения.
enum ThinkParser {
    static func parse(_ text: String, reasoning: Bool) -> ParsedReply {
        let open = "<think>"
        let close = "</think>"

        if let r = text.range(of: close) {
            var think = String(text[..<r.lowerBound])
            if let o = think.range(of: open) {
                think.removeSubrange(think.startIndex..<o.upperBound)
            }
            let answer = String(text[r.upperBound...])
            return ParsedReply(thinking: trim(think), answer: trim(answer), thinkingDone: true)
        }
        if text.hasPrefix(open) {
            return ParsedReply(thinking: trim(String(text.dropFirst(open.count))),
                               answer: "", thinkingDone: false)
        }
        if reasoning {
            return ParsedReply(thinking: trim(text), answer: "", thinkingDone: false)
        }
        return ParsedReply(thinking: "", answer: text, thinkingDone: true)
    }

    private static func trim(_ s: String) -> String {
        s.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
