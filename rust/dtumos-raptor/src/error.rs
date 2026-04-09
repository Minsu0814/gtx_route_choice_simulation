use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RaptorError {
    #[error("GTFS loading error: {0}")]
    GtfsLoad(String),

    #[error("CSV parse error: {0}")]
    CsvParse(#[from] csv::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("No transit data loaded")]
    NoData,

    #[error("No access stops found within {0}m of origin")]
    NoAccessStops(f64),

    #[error("No egress stops found within {0}m of destination")]
    NoEgressStops(f64),

    #[error("Invalid time: {0}")]
    InvalidTime(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

impl From<RaptorError> for PyErr {
    fn from(err: RaptorError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}
