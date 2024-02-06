from skfem import *
from skfem.io.json import from_file
from skfem.models.poisson import vector_laplace, mass, laplace
from skfem.models.general import divergence, rot

from pathlib import Path

import numpy as np
ElemertLib = {
              'ElementTriP1': ElementTriP1(),
              'ElementTriP2': ElementTriP2(),
              'ElementTriP3': ElementTriP3(),
              'ElementTriP4': ElementTriP4()
              }

Epsilon = [0, 0.1, 0.001, 1e-5, 1e-7]
# Eps_static = 0.4
#Eps_static = 0.00000062
Eps_static = 0.0000025

#Eps_static = 0
#Eps_static = 0.1
#Eps_static = 0.001
#Eps_static = 1e-4
# Eps_static = 1/500
# nlevels0 = 4
# nlevels =
# h0 = 1.0 / 2**nlevels0

#Eps_static = 1e-3
#Eps_static = 1e-4


#Eps_static = 1e-5
#Eps_static = 1e-7
#Eps_static = 1e-15

for i in range(4):
    for j in range(4):
        # шаг внутри скобки как 1/4
        mesh = MeshTri.init_circle(5) # Создание треугольной сетки в форме круга с 4 элементами
        # Определение элементов, P2 - кусочно-линейные, P1 - линейные
        element = {'u': ElementVector(ElemertLib['ElementTriP' + str(i+1)]),
                   'p': ElemertLib['ElementTriP' + str(j+1)]}
        basis = {variable: Basis(mesh, e, intorder=3)
                 for variable, e in element.items()}

        # Определение силы тела
        @LinearForm
        def body_force(v, w):
            return w.x[0] * v[1]

        # сборка матрицы для векторного оператора Лапласа
        A = asm(vector_laplace, basis['u'])
        # сборка матрицы для оператора дивергенции
        B = asm(divergence, basis['u'], basis['p'])
        # сборка матрицы для оператора массы
        C = asm(mass, basis['p'])
        # сборка блочной матрицы системы уравнений с учетой нулевой вязкости в блоке давления
        # Менять значения с 1e-6 (изменение вязкости)
        K = bmat([[A, -B.T],
                   [-B, Eps_static * C]], 'csr')
        #K = bmat([[A, -B.T],
        #          [-B, 10e20 * C]], 'csr')
        # сбор вектора правой части уравнения
        f = np.concatenate([asm(body_force, basis['u']),
                            basis['p'].zeros()])
        # решение системы линейных уравнений
        uvp = solve(*condense(K, f, D=basis['u'].get_dofs()))
        # разделение решения на компоненты скорости и давления
        velocity, pressure = np.split(uvp, K.blocks)
        # создание нового базиса для поля потенциала psi
        basis['psi'] = basis['u'].with_element(ElementTriP2())
        # сборка матрицы для оператора Лапласа на новом базисе
        A = asm(laplace, basis['psi'])
        # вычисление вихря на новом базисе
        vorticity = asm(rot, basis['psi'], w=basis['u'].interpolate(velocity))
        # решение системы для потенциала psi
        psi = solve(*condense(A, vorticity, D=basis['psi'].get_dofs()))

        # Replace non-finite values in psi with 0
        psi[np.logical_not(np.isfinite(psi))] = 0


        if __name__ == '__main__':

            from os.path import splitext
            from sys import argv

            from matplotlib.tri import Triangulation

            from skfem.visuals.matplotlib import plot, draw, savefig

            name = splitext(argv[0])[0]
            # сохранение данных о скорости в вткшках
            mesh.save(f'{name}_velocity_{'ElementTriP' + str(i+1)}_{'ElementTriP' + str(j+1)}_{Eps_static}.vtk',
                      {'velocity': velocity[basis['u'].nodal_dofs].T})


            ax = draw(mesh)
            # график давления
            plot(basis['p'], pressure, ax=ax)
            # запись в файл
            savefig(f'{name}_pressure__{'ElementTriP' + str(i+1)}_{'ElementTriP' + str(j+1)}_{Eps_static}.png')

            ax = draw(mesh)
            # построение графика векторного поля скорости в файл
            velocity1 = velocity[basis['u'].nodal_dofs]
            ax.quiver(*mesh.p, *velocity1, mesh.p[0])  # colour by buoyancy
            savefig(f'{name}_velocity__{'ElementTriP' + str(i+1)}_{'ElementTriP' + str(j+1)}_{Eps_static}.png')

            ax = draw(mesh)
            # построение контура для потенциала psi
            ax.tricontour(Triangulation(*mesh.p, mesh.t.T),
                          psi[basis['psi'].nodal_dofs.flatten()])
            savefig(f'{name}_stream-function__{'ElementTriP' + str(i+1)}_{'ElementTriP' + str(j+1)}_{Eps_static}.png')
    j+=1
i+=1
from PIL import Image, ImageDraw
import os
import glob
import re

# ... (ваш предыдущий код)
#
# def combine_images(name, eps_static, output_folder='combined_images'):
#     # Создание директории для сохранения результата, если она не существует
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # Определение категорий изображений
#     categories = ['pressure', 'velocity', 'stream-function']
#
#     for category in categories:
#         # Находим все файлы PNG в данной категории
#         pattern = f'{name}_{category}_*.png'
#         image_files = glob.glob(pattern)
#         if not image_files:
#             print(f"No PNG images found for the category '{category}'.")
#             continue
#
#         # Сортировка файлов по порядку индексов в их названии
#         image_files.sort(key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x)])
#
#         # Загрузка изображений и подготовка метаданных
#         images = []
#         for file in image_files:
#             img = Image.open(file)
#             filename = os.path.splitext(os.path.basename(file))[0]
#             images.append((filename, img))
#
#         # Вычисление размеров для объединенной картинки
#         text_height = 20  # Высота текста
#         total_height = sum(image.height + text_height for _, image in images) + text_height  # Дополнительная высота для epsilon
#
#         max_width = max(image.width for _, image in images)
#
#         # Создание нового изображения для данной категории
#         combined_image = Image.new('RGB', (max_width, total_height), "black")
#
#         # Добавление epsilon в начало изображения
#         draw = ImageDraw.Draw(combined_image)
#         eps_text = f'Epsilon: {eps_static}'
#         draw.text((10, 0), eps_text, fill="white")
#         y_offset = text_height
#
#         # Объединение изображений с добавлением названий
#         for filename, image in images:
#             # Добавление названия файла
#             draw = ImageDraw.Draw(combined_image)
#             draw.text((10, y_offset), filename, fill="white")
#             y_offset += text_height
#
#             # Добавление изображения
#             combined_image.paste(image, (0, y_offset))
#             y_offset += image.height
#
#         # Сохранение объединенного изображения для данной категории
#         combined_image.save(os.path.join(output_folder, f'{name}_{category}_{eps_static}_combined.png'))
#
# # Пример вызова функции
# combine_images('main', Eps_static)
def combine_images(name, eps_static, output_folder='combined_images'):
    # Создание директории для сохранения результата, если она не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Определение категорий изображений
    categories = ['pressure', 'velocity', 'stream-function']

    # Список для хранения объединенных изображений каждой категории
    combined_images = []

    for category in categories:
        # Находим все файлы PNG в данной категории
        pattern = f'{name}_{category}_*.png'
        image_files = glob.glob(pattern)
        if not image_files:
            print(f"No PNG images found for the category '{category}'.")
            continue

        # Сортировка файлов по порядку индексов в их названии
        image_files.sort(key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x)])

        # Загрузка изображений и подготовка метаданных
        images = []
        for file in image_files:
            img = Image.open(file)
            filename = os.path.splitext(os.path.basename(file))[0]
            images.append((filename, img))

        # Вычисление размеров для объединенной картинки
        text_height = 20  # Высота текста
        total_height = sum(image.height + text_height for _, image in images) + text_height  # Дополнительная высота для epsilon

        max_width = max(image.width for _, image in images)

        # Создание нового изображения для данной категории
        combined_image = Image.new('RGB', (max_width, total_height), "black")

        # Добавление epsilon в начало изображения
        draw = ImageDraw.Draw(combined_image)
        eps_text = f'Epsilon: {eps_static}'
        draw.text((10, 0), eps_text, fill="white")
        y_offset = text_height

        # Объединение изображений с добавлением названий
        for filename, image in images:
            # Добавление названия файла
            draw = ImageDraw.Draw(combined_image)
            draw.text((10, y_offset), filename, fill="white")
            y_offset += text_height

            # Добавление изображения
            combined_image.paste(image, (0, y_offset))
            y_offset += image.height

        # Добавление объединенного изображения в список
        combined_images.append(combined_image)

    # Объединение изображений горизонтально
    total_main_image = Image.new('RGB', (sum(img.width for img in combined_images), max(img.height for img in combined_images)), "black")
    x_offset = 0
    for img in combined_images:
        total_main_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Сохранение объединенного изображения для данной категории
    total_main_image.save(os.path.join(output_folder, f'total_main_{eps_static}_combined.png'))

# Пример вызова функции
combine_images('main', Eps_static)