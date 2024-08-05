# Triangles

Requirements

```
python >= 3.8
torch >= 1.8
```

```
pip3 install ultralytics easyocr pytesseract
```

Пример запуска скрипта

```
python find_numbers.py --photos data/photo4.jpg --test_dir test_new3/photo4 --checkpoint best_3.pt
```

Если хотим по ходу смотреть изображения и дебаг вывод

```
python find_numbers.py --photos data/photo4.jpg --test_dir test_new3/photo4 --checkpoint best_3.pt --imshow
```
