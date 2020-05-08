from pathlib import Path
from typing import List

import pytest

from vcap import CAPSULE_EXTENSION, package_capsule, load_capsule, BaseCapsule
from vcap.testing import perform_capsule_tests

# Retrieve all of the capsule paths in a pytest parametrize friendly way
capsules_dir = Path("capsules")
capsule_paths = sorted([str(unpacked_capsule)
                        for unpacked_capsule in Path(capsules_dir).iterdir()
                        if unpacked_capsule.is_dir()])
capsule_paths_argvals = [(path,) for path in capsule_paths]


@pytest.mark.parametrize(
    argnames=["unpackaged_capsule_dir"],
    argvalues=capsule_paths_argvals,
    ids=capsule_paths)
def test_public_capsules(unpackaged_capsule_dir):
    """Test each capsules using the vcap provided test utilities."""

    image_paths = list(Path("tests/test-resources").glob("*"))
    perform_capsule_tests(
        unpackaged_capsule_dir=unpackaged_capsule_dir,
        image_paths=image_paths)
