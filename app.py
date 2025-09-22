# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from solver import run_solver
import json

app = Flask(__name__)

@app.route('/solve_scp_1d', methods=['POST'])
def solve_endpoint():
    """
    API эндпоинт для решения задачи 1D раскроя.
    Принимает данные в теле POST-запроса и параметры в query string.
    """
    try:
        # Получаем параметры из query string
        stock_length = request.args.get('STOCK_LENGTH', default=12000, type=int)
        saw_kerf = request.args.get('SAW_KERF', default=5, type=int)

        # Получаем данные из тела запроса
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "Request body must contain valid JSON"}), 400

        # Вызываем основную логику расчета
        result = run_solver(input_data, stock_length, saw_kerf)

        # Возвращаем результат
        # Используем json.dumps для поддержки кириллицы, а затем загружаем обратно,
        # чтобы jsonify корректно обработал объект.
        response_data = json.loads(json.dumps(result, ensure_ascii=False))
        return jsonify(response_data)

    except Exception as e:
        # Обработка непредвиденных ошибок
        return jsonify({"error": "An internal error occurred", "details": str(e)}), 500

if __name__ == '__main__':
    # Запуск сервера для локального тестирования
    # В production-среде используйте Gunicorn или другой WSGI сервер
    app.run(host='0.0.0.0', port=8080, debug=True)

