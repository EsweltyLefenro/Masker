import threading
import re
import sys
import os
import queue
from pathlib import Path

# Требуем полностью локальную загрузку модели.
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import tkinter as tk
from tkinter import filedialog, messagebox

from transformers import AutoTokenizer
from transformers.pipelines.token_classification import TokenClassificationPipeline

# Предкомпилированные шаблоны ───────────────────────────────────
_PHONE_RE    = re.compile(r"(?<!\w)(?:\+7|8)?[\s\-]?\(?\d{3,4}\)?[\s\-]?\d{2,3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\w)")
_PASSPORT_RE = re.compile(r"\b\d{4}[-\s]?\d{6}\b|\b\d{2}[-\s]?\d{2}[-\s]?\d{6}\b")
_MONEY_RE    = re.compile(r"\b\d{1,3}(?:[\s ]?\d{3})*(?:[\s ]?(?:руб(?:лей)?|\u20BD))(?!\w)")
_EMAIL_RE    = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
# ───────────────────────────────────────────────────────────────


_ner_pipeline = None

def get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        _ner_pipeline = load_ner_pipeline()
    return _ner_pipeline

def load_ner_pipeline():
    # определяем базовый путь (для frozen)
    base = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).parent
    model_path = base / "model"

    # Токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # Загрузить готовую квантованную модель целиком
    # указываем weights_only=False, чтобы загрузить весь объект
    model = torch.load(model_path / "model_quantized.pt", weights_only=False)

    model.config.output_attentions = False
    model.config.output_hidden_states = False
    model.config.return_dict = True

    # Создаём HF-pipeline
    return TokenClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        framework="pt",
        device=0 if torch.cuda.is_available() else -1
    )


def mask_entities(text, entities):
    """
    Маскируем NER-сущности по исходным позициям, объединяя пересечения.
    Одна сущность может быть найдена в нескольких перекрывающихся окнах.
    """
    repls = []
    for ent in entities:
        label = ent["entity_group"]
        word = ent.get("word", "").lower()

        if label == "ORG":
            if re.search(r"\b(улица|проспект|шоссе|аллея|переулок|площадь)\b", word):
                repl = "<адрес>"
            else:
                continue
        elif label == "PER":
            repl = "<ФИО>"
        elif label in ["LOC", "GPE"]:
            repl = "<адрес>"
        elif label == "MONEY":
            repl = "<сумма>"
        elif label == "DATE":
            repl = "<дата>"
        elif label == "EVENT":
            repl = "<событие>"
        else:
            continue

        start, end = ent["start"], ent["end"]
        if not 0 <= start < end <= len(text):
            continue
        repls.append((start, end, repl))

    # При равном начале сначала обрабатываем более длинный диапазон.
    repls.sort(key=lambda item: (item[0], -item[1]))
    merged = []
    for start, end, repl in repls:
        if merged and start < merged[-1][1]:
            previous_start, previous_end, previous_repl = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end), previous_repl)
        else:
            merged.append((start, end, repl))

    result = []
    last = 0
    for start, end, repl in merged:
        result.append(text[last:start])
        result.append(repl)
        last = end
    result.append(text[last:])
    return "".join(result)

def apply_regex_masking(text):
    text = re.sub(r"[\u200b\u00a0\u202f]", " ", text)
    text = _PHONE_RE.sub("<телефон>", text)
    text = _PASSPORT_RE.sub("<паспорт>", text)
    text = _MONEY_RE.sub("<сумма>", text)
    text = _EMAIL_RE.sub("<email>", text)
    # Маскируем названия улиц, не захватывая следующие предложения.
    # Поддержка: "улица Ленина", "ул. Большая Садовая", "улица 60 лет Октября".
    street_prefix = (
        r"(?i:\b(?:ул\.?|улиц[ауыеёй]|пр\.?|проспект(?:а|е|у|ом)?|пр-кт|пр-т|"
        r"пер\.?|переулок(?:а|е|у|ом)?|пер-ул|пл\.?|площадь|площади|площадью|"
        r"ш\.?|шоссе|алле[яиею]))"
    )
    street_name = (
        r"(?:\d{1,4}(?:[-–][ая])?(?:\s+лет)?\s+)?"
        r"[А-ЯЁ][а-яё-]+(?:\s+[А-ЯЁ][а-яё-]+){0,3}"
    )
    text = re.sub(fr"{street_prefix}\s+{street_name}", "<адрес>", text)

    text = re.sub(r"\b\d{1,2}[./]\d{1,2}[./]\d{4}\b", "<дата>", text)
    # Маскируем даты со словами месяца (пример: 17 августа 1993 года)
    text = re.sub(
        r"\b\d{1,2}\s+(?:январь|января|январе|февраль|февраля|феврале|март|марта|марте|апрель|апреля|апреле|май|мая|мае|июнь|июня|июне|июль|июля|июле|август|августа|августе|сентябрь|сентября|сентябре|октябрь|октября|октябре|ноябрь|ноября|ноябре|декабрь|декабря|декабре)\s+\d{4}\b",
        "<дата>",
        text,
        flags = re.IGNORECASE
    )
    text = re.sub(r"(дом|д\.)\s*\d+[а-я]?", "<адрес>", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+[а-я]?\s*(дом|д\.)", "<адрес>", text, flags=re.IGNORECASE)
    text = re.sub(r"(квартира|кв\.?|кв)\s*\d+[а-я]?", "<адрес>", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+[а-я]?\s*(квартира|кв\.?|кв)", "<адрес>", text, flags=re.IGNORECASE)
    text = re.sub(r"<адрес>\s+\d+[а-я]?", "<адрес>", text)
    text = re.sub(r"\d+[а-я]?\s+<адрес>", "<адрес>", text)
    text = re.sub(r"\b[а-яё]{5,}(ой|ая|ее|ей)\b\s+<адрес>", "<адрес>", text, flags=re.IGNORECASE)
    text = re.sub(r"<адрес>\s+\b[а-яё]{5,}(ой|ая|ее|ей)\b", "<адрес>", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(<адрес>)(?:\s*,\s*|\s+)\d+\b",
        r"\1",
        text
    )
    # Удаляем остатки цифр после <адрес>
    text = re.sub(r"(<адрес>)(?:\s*,\s*|\s+)\d+\b", r"\1", text)
    # Удаляем буквенные остатки после <адрес>
    text = re.sub(r"(<адрес>)[А-ЯЁа-яё]+", r"\1", text)

    # ==== удаляем остатки букв вокруг тегов <ФИО> ====
    text = re.sub(r"[А-ЯЁа-яё]+<ФИО>", "<ФИО>", text)
    text = re.sub(r"<ФИО>[А-ЯЁа-яё]+", "<ФИО>", text)
    # ==== конец удаления остатков вокруг <ФИО> ====
    
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    return text

def compress_duplicates(text):
    text = re.sub(r'(<ФИО>\s*){2,}', '<ФИО> ', text)
    text = re.sub(r'(<адрес>\s*){2,}', '<адрес> ', text)
    return text

def anonymize_text(text: str, ner_pipeline) -> str:
    # Токенизируем весь текст и готовим скользящее окно по токенам, сохраняя оффсеты,
    # чтобы не терять символы-разделители и переводы строк.
    tokenizer = ner_pipeline.tokenizer
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=False,
        add_special_tokens=False
    )
    offsets = encoding["offset_mapping"]
    total_tokens = len(offsets)

    # Параметры для окон: макс. токенов и перекрытие
    max_tokens = 256
    stride = 128
    all_entities = []

    # Скользящее окно по токенам
    for start_token in range(0, total_tokens, max_tokens - stride):
        end_token = min(start_token + max_tokens, total_tokens)
        # вычисляем символьные границы в тексте
        char_start = offsets[start_token][0]
        char_end = offsets[end_token - 1][1]
        chunk_text = text[char_start:char_end]
        # инференс сущностей в чанке
        ents = ner_pipeline(chunk_text)
        # корректируем оффсеты под оригинальный текст
        for e in ents:
            all_entities.append({
                "start": e["start"] + char_start,
                "end": e["end"] + char_start,
                "entity_group": e["entity_group"],
                "word": e["word"]
            })

    # Применяем NER-маскирование по найденным сущностям
    ner_masked = mask_entities(text, all_entities)

    # Динамическая маскировка повторных вхождений частей тех же ФИО
    # Собираем все полные ФИО, найденные NER
    person_names = {
        text[e["start"]:e["end"]]
        for e in all_entities
        if e["entity_group"] == "PER"
    }
    # Для каждого полного ФИО бьем на слова и маскируем каждое (кроме слишком коротких)
    for full_name in person_names:
        for part in full_name.split():
            if len(part) > 2:
                ner_masked = re.sub(
                    rf"\b{re.escape(part)}\b",
                    "<ФИО>",
                    ner_masked
                )

    # Применяем regex-маскирование (телефоны, паспорта и т.д.)
    regex_masked = apply_regex_masking(ner_masked)
    # Убираем дубли тегов и возвращаем результат
    return compress_duplicates(regex_masked)


def run_gui():
    root = tk.Tk()
    root.title("Masker")
    root.geometry("400x200")

    # если у вас есть файл с иконкой рядом с main.py — .png или .ico
    try:
        icon = tk.PhotoImage(file="mask_icon.png")
        root.iconphoto(False, icon)
    except Exception:
        pass

    # Статус и кнопка сразу показываются
    status = tk.Label(root, text="Masker: загрузка модели… Пожалуйста, подождите.", font=("Arial", 12))
    status.pack(pady=10)

    instr = tk.Label(
        root,
        text="Выберите текстовый файл (.txt) ниже. Замаскированная копия (так же .txt) сохранится рядом с тем же именем, но с постфиксом -masked.",
        wraplength=380,
        font=("Arial", 10),
        justify="center"
    )
    instr.pack(pady=(0,10))

    button = tk.Button(root, text="Выбрать файл", state="disabled", font=("Arial", 12))
    button.pack()

    def select_file():
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            pipeline = get_ner_pipeline()
            result = anonymize_text(text, pipeline)
            input_path = Path(file_path)
            output_path = input_path.with_name(f"{input_path.stem}-masked.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
            messagebox.showinfo("Успех", f"Маскированный файл сохранён как:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    button.config(command=select_file)

    # Передаём результат из фонового потока в главный поток Tkinter.
    model_load_results = queue.Queue()

    def load_model_background():
        try:
            get_ner_pipeline()
        except Exception as exc:
            model_load_results.put((False, str(exc)))
        else:
            model_load_results.put((True, None))

    def on_model_loaded():
        status.config(text="Модель загружена. Нажмите «Выбрать файл» ниже.")
        button.config(state="normal")

    def check_model_load():
        try:
            success, error = model_load_results.get_nowait()
        except queue.Empty:
            root.after(100, check_model_load)
            return

        if success:
            on_model_loaded()
        else:
            status.config(text="Ошибка загрузки модели.")
            messagebox.showerror("Ошибка загрузки модели", error)

    threading.Thread(target=load_model_background, daemon=True).start()
    root.after(100, check_model_load)

    root.mainloop()

if __name__ == "__main__":
    run_gui()
