import json

# 读取文件内容
with open('data/en_int.json', 'r') as file:
    content = file.read()

# 将内容拆分成行并封装成一个数组
content = "[" + ",".join(content.strip().split('\n')) + "]"

# 载入修正后的JSON内容
json_data = json.loads(content)

# 提取前200个元素
extracted_data = []
for item in json_data:  # Change from 50 to 200
    extracted_data.append({
        "query": item["query"],
        "answer": item["answer"],
        "positive": item["positive"]
    })

instruction = "Write a high-quality answer for the given question using only the following relevant search results, please answer in as much detail as possible based on chunk, no generalisations!"

# 存储所有prompt的列表
prompts = []

# 生成prompt
for entry in extracted_data:
    question = entry['query']
    positives = entry['positive']
    
    # 从positives创建chunks，移除方括号
    chunks = "\n".join([f"###Chunk {i+1}: {' '.join(positive).replace('[', '').replace(']', '')}" for i, positive in enumerate(positives)])
    
    prompt = (
        f"###Instruction: {instruction}\n"
        f"{chunks}\n"
        f"###Question: {question}\n"
    )
    
    # 添加prompt到列表
    prompts.append({
        "prompt": prompt,
        "answer": entry["answer"]
    })

# 打印生成的prompts和answers
# for x in prompts:
#     print(x["prompt"])
#     print(x["answer"])

# 检查生成的回答是否存在于对应的answer中
def is_answer_correct(generated_answer, expected_answers):
    # 如果expected_answers是一个列表列表
    if isinstance(expected_answers[0], list):
        for sublist in expected_answers:
            if generated_answer in sublist:
                return True
  
    elif isinstance(expected_answers, list):
        if generated_answer in expected_answers:
            return True
    return False

# 示例: 假设大模型生成了一些回答
generated_answers = [
    "January 2 2022",  # Example for a correct answer
    "Jan. 2, 2022",    # Example for another correct answer
    "wrong answer"     # Example for an incorrect answer
]

generated_answers2 = "January 2 2022"
is_correct = is_answer_correct(generated_answers2, prompts[0]["answer"])
# print(is_correct)

# # 假设我们对第一个prompt进行验证
# for generated_answer in generated_answers:
#     is_correct = is_answer_correct(generated_answer, prompts[0]["answer"])
#     print(f"Generated Answer: {generated_answer} - Correct: {is_correct}")

# # 如果要检查所有prompt，可以如下：
# for i, prompt_entry in enumerate(prompts):
#     for generated_answer in generated_answers:
#         is_correct = is_answer_correct(generated_answer, prompt_entry["answer"])
#         print(f"Prompt {i+1}, Generated Answer: {generated_answer} - Correct: {is_correct}")
