project(
  'framework',
  'cpp',
  version: '0.1',
  default_options: ['warning_level=3', 'cpp_std=c++20'],
)

add_project_arguments('-Wno-sign-compare', language: 'cpp')

dep_fmt = dependency('fmt', required: true)
dep_spdlog = dependency('spdlog', required: true)
dep_gtest = dependency('gtest', required: true)

deps = [dep_fmt, dep_spdlog, dep_gtest]
incdir = include_directories('src')
srcs = ['src/logger.cpp']

all_tests = ['env', 'logger', 'tensor', 'module']

foreach t : all_tests
  exe = executable(
    'test_' + t,
    ['tests/test_' + t + '.cpp'] + srcs,
    include_directories: incdir,
    dependencies: deps,
  )
  test('test_' + t, exe)
endforeach

# py = import('python').find_installation('python3', required: true)
# pybind11_incdir = py.get_install_dir() / 'pybind11' / 'include'
# message('Python include dir: ' + pybind11_incdir)
# pybind11_dep = dependency('pybind11', required: true)
# message('Python dependency: ' + pybind11_dep.name())
