import pytest
from unittest.mock import patch, MagicMock
from rgt_digital_twin.literature_extraction import main, process_single_pdf
import json

@patch('rgt_digital_twin.literature_extraction.genai.Client')
def test_main_no_pdf_files_found(mock_genai_client):
    with patch('pathlib.Path.glob', return_value=[]), \
         patch('builtins.open', MagicMock()), \
         patch('json.load', return_value={}), \
         patch('sys.exit') as mock_exit:
        main(folder_path="/some/folder", entities_file_path="entities.json", output_filename="output.json")
        mock_exit.assert_called_once_with(1)

@patch('rgt_digital_twin.literature_extraction.genai.Client')
@patch('builtins.open', new_callable=MagicMock)
@patch('json.loads')
def test_process_single_pdf_successful(mock_json_loads, mock_open, mock_genai_client):
    # Arrange
    mock_pdf_path = MagicMock()
    mock_pdf_path.name = "test.pdf"
    entities_to_extract = {"entity1": "instruction1"}

    # Configure the mock for open to simulate reading bytes from a file
    mock_file_handle = MagicMock()
    mock_file_handle.read.return_value = b"fake pdf content"
    mock_open.return_value.__enter__.return_value = mock_file_handle

    mock_genai_client.return_value.models.generate_content.return_value.text = '{"entity1": "value1"}'
    mock_json_loads.return_value = {"entity1": "value1"}

    # Act
    result = process_single_pdf(mock_pdf_path, entities_to_extract, mock_genai_client)

    # Assert
    assert result == {"entity1": "value1"}

@patch('rgt_digital_twin.literature_extraction.genai.Client')
@patch('builtins.open', new_callable=MagicMock)
def test_process_single_pdf_api_error(mock_open, mock_genai_client):
    # Arrange
    mock_pdf_path = MagicMock()
    mock_pdf_path.name = "test.pdf"
    entities_to_extract = {"entity1": "instruction1"}

    # Configure the mock for open to simulate reading bytes from a file
    mock_file_handle = MagicMock()
    mock_file_handle.read.return_value = b"fake pdf content"
    mock_open.return_value.__enter__.return_value = mock_file_handle

    mock_genai_client.return_value.models.generate_content.side_effect = Exception("API error")

    # Act
    result = process_single_pdf(mock_pdf_path, entities_to_extract, mock_genai_client)

    # Assert
    assert result == {"entity1": "ERROR: API call failed"}

@patch('rgt_digital_twin.literature_extraction.genai.Client')
@patch('builtins.open', new_callable=MagicMock)
@patch('json.loads')
def test_process_single_pdf_json_error(mock_json_loads, mock_open, mock_genai_client):
    # Arrange
    mock_pdf_path = MagicMock()
    mock_pdf_path.name = "test.pdf"
    entities_to_extract = {"entity1": "instruction1"}

    # Configure the mock for open to simulate reading bytes from a file
    mock_file_handle = MagicMock()
    mock_file_handle.read.return_value = b"fake pdf content"
    mock_open.return_value.__enter__.return_value = mock_file_handle

    mock_genai_client.return_value.models.generate_content.return_value.text = "invalid json"
    mock_json_loads.side_effect = json.JSONDecodeError("msg", "doc", 0)

    # Act
    result = process_single_pdf(mock_pdf_path, entities_to_extract, mock_genai_client)

    # Assert
    assert result == {"entity1": "ERROR: JSONDecodeError"}
