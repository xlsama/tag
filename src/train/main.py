"""训练主脚本"""

import json

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

from . import config


def load_labels() -> list[str]:
    """加载标签列表"""
    data = json.loads(config.LABELS_FILE.read_text(encoding='utf-8'))
    return [label['name'] for label in data['labels']]


def load_data() -> list[dict]:
    """加载训练数据"""
    lines = config.DATA_FILE.read_text(encoding='utf-8').strip().split('\n')
    return [json.loads(line) for line in lines if line.strip()]


def format_prompt(text: str, labels: list[str], all_labels: list[str]) -> str:
    """将数据转换为对话格式"""
    labels_str = ', '.join(all_labels)
    output = json.dumps(labels, ensure_ascii=False)

    return f"""<|im_start|>system
你是一个评论标签分类助手。请从以下标签中选择所有适用的标签，以 JSON 数组格式返回。
可选标签：{labels_str}<|im_end|>
<|im_start|>user
请为以下评论打标签：
{text}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""


def prepare_dataset(data: list[dict], all_labels: list[str]) -> Dataset:
    """准备训练数据集"""
    formatted = [{'text': format_prompt(item['text'], item['labels'], all_labels)} for item in data]
    return Dataset.from_list(formatted)


def train():
    """训练模型"""
    print('=' * 50)
    print('开始训练评论多标签分类模型')
    print('=' * 50)

    # 加载数据
    all_labels = load_labels()
    data = load_data()
    dataset = prepare_dataset(data, all_labels)
    print(f'标签: {all_labels}')
    print(f'样本数: {len(data)}')

    # 加载模型
    print(f'加载模型: {config.MODEL_NAME}')
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # 添加 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.TARGET_MODULES,
        use_gradient_checkpointing='unsloth',
    )

    # 训练
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = SFTConfig(
        output_dir=str(config.OUTPUT_DIR),
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        save_total_limit=2,
        bf16=True,
        optim='adamw_8bit',
        report_to='none',
        dataset_text_field='text',
        max_length=config.MAX_SEQ_LENGTH,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,  # type: ignore
    )

    print('开始训练...')
    trainer.train()

    # 保存模型
    lora_path = config.OUTPUT_DIR / 'lora_adapter'
    merged_path = config.OUTPUT_DIR / 'merged_model'

    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f'LoRA 适配器已保存: {lora_path}')

    model.save_pretrained_merged(str(merged_path), tokenizer, save_method='merged_16bit')
    print(f'合并模型已保存: {merged_path}')

    print('=' * 50)
    print('训练完成!')
    print('=' * 50)


if __name__ == '__main__':
    train()
