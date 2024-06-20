use numpy::ndarray::{Array4, ArrayView4, ArrayViewMut4, Axis};
use pyo3::{ pymodule, types::{ PyModule }, Bound, PyResult };

#[pymodule]
fn pylib<'py>(m: &Bound<'py, PyModule>) -> PyResult<PyObject> {
    #[pyfn(m)]
    #[pyo3(name = "create")]
    fn create<'py>(shape: Vec<usize>, ) -> PyResult<PyObject> {
        assert!(shape.len() == 4);
    }
}
