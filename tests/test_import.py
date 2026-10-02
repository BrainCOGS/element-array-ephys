"""Import smoke tests that run without a database or the `elements` extra."""

import importlib

import pytest

# Modules whose import needs only the core dependencies (plus element-interface,
# which every ephys module uses). Left out on purpose: `export.nwb` needs the
# `nwb` extra, and `readers.kilosort_triggering` needs the Allen
# ecephys_spike_sorting / pykilosort packages.
MODULES = [
    "element_array_ephys",
    "element_array_ephys.version",
    "element_array_ephys.probe",
    "element_array_ephys.ephys_acute",
    "element_array_ephys.ephys_chronic",
    "element_array_ephys.ephys_no_curation",
    "element_array_ephys.ephys_precluster",
    "element_array_ephys.ephys_report",
    "element_array_ephys.readers.kilosort",
    "element_array_ephys.readers.openephys",
    "element_array_ephys.readers.probe_geometry",
    "element_array_ephys.readers.spikeglx",
    "element_array_ephys.readers.utils",
    "element_array_ephys.plotting.corr",
    "element_array_ephys.plotting.probe_level",
    "element_array_ephys.plotting.qc",
    "element_array_ephys.plotting.unit_level",
    "element_array_ephys.plotting.widget",
]


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    importlib.import_module(module)


def test_version_is_set():
    from element_array_ephys.version import __version__

    assert __version__
