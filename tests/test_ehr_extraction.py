import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from rgt_digital_twin.ehr_extraction import main, process_pdf
import asyncio

@pytest.mark.asyncio
async def test_main_no_pdf_files_found():
    with patch('pathlib.Path.glob', return_value=[]), \
         patch('builtins.open', MagicMock()), \
         patch('json.load', return_value={}), \
         patch('sys.exit') as mock_exit:
        await main(folder_path="/some/folder", entities_file_path="entities.json", output_filename="output.json")
        mock_exit.assert_called_once_with(1)

@pytest.mark.asyncio
async def test_process_pdf_successful():
    # Arrange
    mock_pdf_path = MagicMock()
    mock_pdf_path.name = "test.pdf"
    entities_to_extract = {"entity1": "instruction1"}
    mock_model = MagicMock()

    with patch('rgt_digital_twin.ehr_extraction.load_pdf_page_contents', new_callable=AsyncMock) as mock_load_pdf, \
         patch('rgt_digital_twin.ehr_extraction.extract_entity_from_page_content', return_value='{"entity1":"value1"}') as mock_extract, \
         patch('rgt_digital_twin.ehr_extraction.select_entity_from_extract', return_value='{"entity1":"value1"}') as mock_select:

        mock_load_pdf.return_value = ["page 1 content"]

        # Act
        result = await process_pdf(mock_pdf_path, entities_to_extract, mock_model)

        # Assert
        mock_load_pdf.assert_called_once_with(str(mock_pdf_path))
        mock_extract.assert_called_once()
        mock_select.assert_called_once()
        assert result == {"entity1": '{"entity1":"value1"}'}

@pytest.mark.asyncio
async def test_process_pdf_load_error():
    # Arrange
    mock_pdf_path = MagicMock()
    mock_pdf_path.name = "test.pdf"
    entities_to_extract = {"entity1": "instruction1"}
    mock_model = MagicMock()

    with patch('rgt_digital_twin.ehr_extraction.load_pdf_page_contents', new_callable=AsyncMock) as mock_load_pdf:
        mock_load_pdf.side_effect = Exception("PDF load error")

        # Act
        result = await process_pdf(mock_pdf_path, entities_to_extract, mock_model)

        # Assert
        assert result == {}
