class PromptAnalyzer:
    def __init__(self, tokenizer, prompt : str):
        self.tokenizer = tokenizer
        self.prompt = prompt
        
        self.tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

    def encode(self, text : str):
        tokens = self.tokenizer.tokenize(text)
        encoded = self.tokenizer.convert_tokens_to_ids(tokens)
        return encoded

    def calc_word_indecies(self, word : str, limit : int = -1, start_pos = 0):
        word = word.lower()
        merge_idxs = []

        tokens = self.tokens
        needles = self.encode(word)

        limit_count = 0
        current_pos = 0
        for i, token in enumerate(tokens):
            current_pos = i
            if i < start_pos:
                continue

            if needles[0] == token and len(needles) > 1:
                next = i + 1
                success = True
                for needle in needles[1:]:
                    if next >= len(tokens) or needle != tokens[next]:
                        success = False
                        break
                    next += 1

                # append consecutive indexes if all pass
                if success:
                    merge_idxs.extend(list(range(i, next)))
                    if limit > 0:
                        limit_count += 1
                        if limit_count >= limit:
                            break

            elif needles[0] == token:
                merge_idxs.append(i)
                if limit > 0:
                    limit_count += 1
                    if limit_count >= limit:
                        break

        return merge_idxs, current_pos