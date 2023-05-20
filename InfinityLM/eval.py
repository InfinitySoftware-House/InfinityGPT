
class EvaluateInfinityLM():
    def eval(target, input, sentence_index):
        eval_loss = 0
        target_split = target[sentence_index][0].split()
        target_split.remove("<endoftext>")
        total_value = len(target_split)
        input = input.split()
        for i, word in enumerate(input):
            if i >= len(target_split):
                break
            if target_split[i] == word:
                eval_loss += 1
        return (eval_loss / total_value)*100