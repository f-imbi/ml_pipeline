from pathlib import Path

import pytest

from kedro_pipeline.run import ProjectContext


@pytest.fixture
def project_context():
    return ProjectContext(str(Path.cwd()))


class TestProjectContext:
    def test_project_name(self, project_context):
        assert project_context.project_name == "kedro_pipeline"

    def test_project_version(self, project_context):
        assert project_context.project_version == "0.16.4"
