# Задания для практической части курса

## 1. Метод прогонки для трехдиагональных матриц
#### Задание
Запрограммировать метод Томаса или любой другой метод для эффективного решения систем линейных уравнений с трехдиагональными матрицами
```math
Ax = b
```
Матрица $`A`$ должна храниться в компактном формате, т.е. использовать примерно $`3 \times N`$ элементов памяти. Использование плотных форматов с объемом $`N \times N`$ не допускается. Использование разреженных форматов, например, CSR, не рекомендуется.

Оформить программу в виде функции, например, следуюшего вида:
```python
x = solve3diag(N, d, du, dl, b)
```
где `N` – размер матрицы, `d` – массив с элементами на главной диагонали, `du` и `dl` – массивы с элементами на верхней и нижней диагоналях, соответственно, `b` – массив с элементами правой части. Возвращаемое значение – массив с элементами решения.

#### Проверка
Для проверки задания необходимо также запрограммировать впомогательную функцию для умножения трехдиагональной матрицы на вектор $`y = Ax`$, например:
```python
y = mul3diag(N, d, du, dl, x)
```
С помощью этой функции вычислить невязку системы $`Ax-b`$ и вычислить её $`L^2`$-норму.

Проверить, что норма невязки близка к нулю на простых и случайных матрицах. Пример для Python с использованием Numpy:
```python
import numpy as np
def solve3diag(N, d, du, dl, b):
    #...
def mul3diag(N, d, du, dl, x):
    #...

# простая матрица
N = 100
d = np.full(N, 2)
du = np.full(N, -1)
dl = np.full(N, -1)
b = np.full(N, 1)
x = solve3diag(N, d, du, dl, b)
res = mul3diag(N, d, du, dl, x) - b
print("L2-norm: ", np.linalg.norm(res))

# случайная матрица
N = 100
d = 2+np.random.rand(N) # d от 2 до 3
du = -np.random.rand(N) # du от -1 до 0
dl = -np.random.rand(N) # dl от -1 до 0
b = 2*np.random.rand(N)-1 # b от -1 до 1
x = solve3diag(N, d, du, dl, b)
res = mul3diag(N, d, du, dl, x) - b
print("L2-norm: ", np.linalg.norm(res))
```

#### Дополнительное необязательное задание
Запрограммируйте решение с помощью метода конечных разностей уравнения Лапласа $`-\frac{d^2 u}{dx^2}=f(x)`$ на отрезке [0,1] с точным решением $`g=\sin \Pi x`$ и граничными условиями типа Дирихле.

Сведите с помощью метода конечных разностей уравнение Лапласа к системе линейных уравнений с трехдиагональной матрицей и решите её с помощью метода прогонки. Используя известный аналитический вид точного решения, оцените скорость сходимости численного решения в $`L^2`$ или $`C`$ норме. 

## 2. Стационарное уравнение конвекции-диффузии
#### Задание
Запрограммировать решение стационарного уравнения конвекции-диффузии на отрезке [0,1] с помощью метода конечных разностей, конечных элементов, или конечных объемов.
```math
Pe \frac{du}{dx} - \frac{d^2u}{dx^2} = 0 \quad \text{ в } [0,1]
```
```math
u(0) = 0, \quad u(1) = 1
```
Для этой задачи известно точное решение:
```math
u(x) = \frac{e^{Pe \, x} - 1}{e^{Pe} - 1}
```

При использовании МКР, МКЭ или МКО нужно свести задачу к системе линейных уравнений с трехдиагональной матрицей и решить её методом прогонки.

Необходимо реализовать два способа аппроксимации конвективного слагаемого – второго и первого порядка точности и сравнить центральные разности со смещенными (для МКР), влияние SUPG (для МКЭ), и противопотоковой аппроксимации (для МКО).

#### Отчет
В отчет желательно включить следующую информацию:
- постановка задачи и вид точного решения
- краткое описание численной схемы и общий вид системы линейных уравнений
- вид полученного численного решения и сравнение с точным решением (показать пример хорошего решения и плохого решения)
- анализ выполнения принципа максимума (проверка, что численное решение не меньше 0 и не больше 1)
- анализ скорости сходимости (например, в виде таблицы с $`L^2`$ или $`C`$ нормами в зависимости от шага сетки для разных схем и для разных коэффициентов Pe)
- общие выводы: отличия использованных схем, оценка их скорости сходимости при разных параметрах Pe, рекомендации по их использованию

#### Дополнительное необязательное задание
Решить нестационарное уравнения конвекции-диффузии на отрезке [0,1]:
```math
\frac{\partial u}{\partial t} + Pe \frac{\partial u}{\partial x} - \frac{\partial^2u}{\partial x^2} = 0 \quad \text{ в } [0,1]
```
```math
u(0) = 1, \quad u(1) = 0
```
Проследите за теми же эффектами, связанными с нарушением принципа максимума.

## 3. Задача Стокса и LBB-условия
Для выполнения задания рекомендуется использование конечно-элементной библиотеки [scikit-fem](https://github.com/kinnala/scikit-fem). Установите эту библиотеку и ознакомьтесь с примерами и документацией.

#### Задание
На основе примера [ex18.py](https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex18.py) исследуйте поведение различных пар базисных элементов для скорости и давления в задаче Стокса для круглой области $`\Omega = \{(x,y): x^2 + y^2 \le 1\}`$:
```math
- \Delta \mathbf{u} + \nabla p = \mathbf{f} \quad \text{ в } \Omega
```
```math
\nabla \cdot \mathbf{u} = 0 \quad \text{ в } \Omega
```
с граничными условиями $`\mathbf{u} = 0`$ на $`\partial\Omega`$ и правой частью $`\mathbf{f}(x,y) = (0, x)`$.

Дополнительно исследуйте, как наличие регуляризации и её тип влияют на устойчивость численной схемы.

Попробуйте изменить форму расчётной области и вид правой части $`\mathbf{f}`$.

Проверьте, есть ли сходимость численного решения при устремлении коэффициента регуляризации $`\varepsilon \to 0`$.

#### Отчет
Результаты исследований оформите в виде краткого отчета:
- перечислите неудачные пары базисных элементов и поясните, почему они не подходят
- перечислите удачные пары базисных элементов и обоснуйте их выбор

#### Дополнительное необязательное задание
Проверьте сходимость численного решения к точному решению. Для этого вы можете сами выбрать точное решение в виде некоторого бездивергентного поля $`\mathbf{u}`$ и модифицировать граничные условия и правую часть $`\mathbf{f}`$ в задаче Стокса. При проверке обратите внимание на влияние регуляризации и выбор пар базисных элементов.

## 4. Линейная упругость и locking-эффекты
Для выполнения задания рекомендуется использование конечно-элементной библиотеки [scikit-fem](https://github.com/kinnala/scikit-fem). Установите эту библиотеку и ознакомьтесь с примерами и документацией.

#### Задание
На основе примеров [shear_lock.py](code/shear_lock.py) и [volume_lock.py](code/volume_lock.py) исследуйте поведение численной модели в двух задачах линейной упругости, предложенных в [главе 8.6](https://solidmechanics.org/Text/Chapter8_6/Chapter8_6.php) из книги Allan F. Bower, Applied Mechanics of Solids.

Дополнительно исследуйте, как тип базисных функций, тип сеточных ячеек и шаг сетки влияют на численное решение.

#### Отчет
Результаты исследований оформите в виде краткого отчета:
- для задачи с балкой укажите, при каких шагах сетки по осям X и Y удаётся избавиться от блокировки
- для задачи с цилиндром укажите, при каких значениях $`\mu`$ возникают эффекты блокировки, и есть ли при этом зависимость от шага сетки
- краткие выводы о возможности избавиться от блокировок с помощью изменения типа базисных функций или типа сеточных ячеек (треугольники вместо четырехугольников)

#### Дополнительное необязательное задание
Попробуйте реализовать идеи, предложенные в [главе 8.6](https://solidmechanics.org/Text/Chapter8_6/Chapter8_6.php) для исправления locking-эффектов.

## 5. Задача о каверне и метод SIMPLE
#### Задание
Запрограммируйте решение задачи о каверне с использованием метода SIMPLE (Semi-Implicit Method for Pressure Linked Equations, Patankar & Spalding, 1972).

Течение несжимаемой вязкой жидкости в единичном квадрате $`\Omega`$ описывается нестационарными уравнениями Навье-Стокса:
```math
	\frac{\partial \mathbf{u}}{\partial t} + \left( \mathbf{u} \cdot \nabla \right) \mathbf{u}
	- \nu \Delta \mathbf{u}
	+ \nabla p = 0 \\
	\nabla \cdot \mathbf{u} = 0
```
Нормальные компоненты скорости на границе равны нулю. На верхней границе области задана постоянная скорость $`\mathbf{u}=(1,0)`$. В начальный момент времени $`\mathbf{u} = 0`$.

Для решения используйте разнесенные сетки и метод итераций SIMPLE для нахождения бездивергентного поля $`\mathbf{u}`$. При дискретизации уравнений можно использовать явную или неявную схему метода конечных объемов. Для аппроксимации потоков через грани ячеек можно использовать центральную схему или противопотоковую. При необходимости можно добавить параметры релаксации в метод SIMPLE: $`\alpha = 0.5, \alpha_p = 0.8`$. Внутренние итерации метода SIMPLE следует останаливать при достижении $`\|\mathrm{div}_h \mathbf{u}_h \| < \varepsilon_1`$. Расчет следует останавливать при установлении стационарного течения, например при $`\|\mathbf{u}^{n+1}-\mathbf{u}^n\| < \varepsilon_2`$.

Шаг сетки по осям x и y для удобства считаем одинаковым. Параметры $`\varepsilon_1`$ и $`\varepsilon_2`$, шаг по времени и шаг по пространству выбираются экспериментально из соображений устойчивости решения.

#### Результаты
Проведите расчеты при разных значениях числа Рейнольдса (за счет изменения параметра $`\nu`$) и проследите за тем, как меняется положение центра главного вихря и удаётся ли увидеть второстепенные вихри в нижних углах области. Качественно сравните ваши результаты с результатами из литературы (ключевые слова для поиска – lid driven cavity flow).

Сделайте выводы об устойчивости вашей схемы при изменении шагов по времени и по пространству.
