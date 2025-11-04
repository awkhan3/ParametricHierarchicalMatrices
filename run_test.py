import runpy
import sys


H_matrix_test_lst = [
    "TestPHSizeScale",
    "TestHSizeScale",
    "TestPHErrorScale",
]
H2_matrix_test_lst = [
    "TestPH2SizeScale",
    "TestH2SizeScale",
]
Extended_H_matrix_test_lst = [
    "TestExtendedHSizeScale",
    "TestExtendedPHSizeScale",
]

for test in H_matrix_test_lst:
    with open('Logs/' + test + '.txt', 'w') as sys.stderr:
        runpy.run_path(path_name = 'ParamHMatrixTests/' + test + '.py')

for test in H2_matrix_test_lst:
    with open('Logs/' + test + '.txt', 'w') as sys.stderr:
        runpy.run_path(path_name = 'ParamH2MatrixTests/' + test + '.py')

for test in Extended_H_matrix_test_lst:
    with open('Logs/' + test + '.txt', 'w') as sys.stderr:
        runpy.run_path(path_name = 'ParamHMatrixTests/' + test + '.py')
