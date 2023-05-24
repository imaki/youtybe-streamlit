import mojimoji

def convert_katakana(text):
    return mojimoji.han_to_zen(text, kana=False)

input_text = "アイウエオ"
output_text = convert_katakana(input_text)
print(output_text)
