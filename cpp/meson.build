project(
  'framework',
  'cpp',
  version: '0.1',
  default_options: ['warning_level=3', 'cpp_std=c++20'],
)

# add_project_arguments('-Wno-sign-compare', language: 'cpp')

if get_option('buildtype').contains('debug')
  add_project_arguments('-UNDEBUG', language: 'cpp')
endif

dep_fmt = dependency('fmt', required: true)
dep_spdlog = dependency('spdlog', required: true)
dep_gtest = dependency('gtest', required: true)

deps = [dep_fmt, dep_spdlog, dep_gtest]
incdir = include_directories('src')
srcs = ['src/logger.cpp', 'src/tensor.cpp']

all_tests = [
  'env',
  'logger',
  'tensor',
  'module',
  'elem_wise',
  'matmul',
  'activation',
  'norm',
  'rope',
  'sampling',
  'softmax',
  'transpose',
  'sdpa',
]

foreach t : all_tests
  exe = executable(
    'test_' + t,
    ['tests/test_' + t + '.cpp'] + srcs,
    include_directories: incdir,
    dependencies: deps,
  )
  test('test_' + t, exe)
endforeach