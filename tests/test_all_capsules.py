from pathlib import Path

import pytest

from vcap import CAPSULE_EXTENSION, package_capsule, load_capsule
from vcap.testing import perform_capsule_tests

# Retrieve all of the capsule paths in a pytest parametrize friendly way
capsules_dir = Path("capsules")
capsule_paths = sorted([str(unpacked_capsule)
                        for unpacked_capsule in Path(capsules_dir).iterdir()
                        if unpacked_capsule.is_dir()])
capsule_paths_argvals = [(Path(path),) for path in capsule_paths]


@pytest.mark.parametrize(
    argnames=["unpackaged_capsule_dir"],
    argvalues=capsule_paths_argvals,
    ids=capsule_paths)
def test_capsules(unpackaged_capsule_dir: Path):
    """Test each capsules using the vcap provided test utilities."""

    image_paths = list(Path("tests/test_resources").glob("*"))
    perform_capsule_tests(
        unpackaged_capsule_dir=unpackaged_capsule_dir,
        image_paths=image_paths)


@pytest.mark.parametrize(
    argnames=["unpackaged_capsule_dir"],
    argvalues=capsule_paths_argvals,
    ids=capsule_paths)
def test_capsule_meets_basic_standards(unpackaged_capsule_dir: Path):
    """Ensures a capsule meets basic repository rules:
        1) The capsule name matches it's directory name
        2) Has a description
    """
    packaged_capsule_path = (unpackaged_capsule_dir
                             .with_name(unpackaged_capsule_dir.stem)
                             .with_suffix(CAPSULE_EXTENSION))
    package_capsule(unpackaged_capsule_dir, packaged_capsule_path)
    capsule = load_capsule(
        packaged_capsule_path,
        unpackaged_capsule_dir,
        inference_mode=False)

    assert capsule.name == unpackaged_capsule_dir.name, \
        f"The capsule directory '{unpackaged_capsule_dir}' does not match the " \
        f"capsules name '{capsule.name}'"
    assert len(capsule.description) > 10, \
        f"The capsule description '{capsule.description}' is too short!"


@pytest.mark.parametrize(
    argnames=["unpackaged_capsule_dir"],
    argvalues=capsule_paths_argvals,
    ids=capsule_paths)
def test_capsules_required_information_files(unpackaged_capsule_dir: Path):
    """Test that each capsule has a certain required files for humans.
    All required runtime files are tested by test_capsules.
    """

    # Verify a licenses directory exists
    licenses_dir = unpackaged_capsule_dir / "licenses"
    assert licenses_dir.is_dir()

    # Verify there is a code license
    code_license = licenses_dir / "code.LICENSE"
    assert code_license.is_file()

    # Verify there is a README.md
    readme_file = unpackaged_capsule_dir / "README.md"
    assert readme_file.is_file()

    # Check if there are model files and verify there is a model license file
    filetypes = ("*.pb", "*.tflite", "*.bin", "*.h5")
    globs = [list(unpackaged_capsule_dir.glob(ftype)) for ftype in filetypes]
    model_files = sum(globs, [])
    if len(model_files):
        model_license = licenses_dir / "model.LICENSE"
        assert model_license.is_file()
