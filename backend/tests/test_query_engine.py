import pytest
from unittest.mock import MagicMock
from app.services.query_engine import QueryEngine


@pytest.fixture
def llm_service_mock():
    mock = MagicMock()
    mock.query.return_value = "This is a test response about CVs"
    return mock


@pytest.fixture
def cv_database_mock():
    mock = MagicMock()
    mock.get_all_cvs.return_value = {
        "cv1": {
            "personal_info": {"name": "John Doe"},
            "skills": ["Python", "Machine Learning"],
        },
        "cv2": {
            "personal_info": {"name": "Jane Smith"},
            "skills": ["JavaScript", "React"],
        },
    }
    return mock


@pytest.fixture
def query_engine(llm_service_mock, cv_database_mock):
    return QueryEngine(llm_service_mock, cv_database_mock)


def test_process_query_simple(query_engine, llm_service_mock, cv_database_mock):
    """Test processing a simple query without context"""
    result = query_engine.process_query("Who has Python skills?")

    # Verify database was accessed
    cv_database_mock.get_all_cvs.assert_called_once()

    # Verify LLM was called with proper prompt
    llm_service_mock.query.assert_called_once()
    prompt = llm_service_mock.query.call_args[0][0]
    assert "Who has Python skills?" in prompt
    assert "John Doe" in prompt
    assert "Jane Smith" in prompt

    # Verify result
    assert result == "This is a test response about CVs"


def test_process_query_with_context(query_engine, llm_service_mock, cv_database_mock):
    """Test processing a query with conversation context"""
    # First query
    query_engine.process_query(
        "Who has Python skills?", user_id="user1", conversation_id="conv1"
    )

    # Second query with context
    result = query_engine.process_query(
        "What other skills does this person have?",
        user_id="user1",
        conversation_id="conv1",
    )

    # Verify context was used in second query
    assert len(llm_service_mock.query.call_args_list) == 2
    second_prompt = llm_service_mock.query.call_args_list[1][0][0]
    assert "Who has Python skills?" in second_prompt
    assert "What other skills does this person have?" in second_prompt

    # Verify result
    assert result == "This is a test response about CVs"


def test_conversation_context_management(query_engine):
    """Test that conversation context is properly managed"""
    # First conversation
    query_engine.process_query("Query 1", user_id="user1", conversation_id="conv1")
    query_engine.process_query("Query 2", user_id="user1", conversation_id="conv1")

    # Second conversation
    query_engine.process_query("Query A", user_id="user2", conversation_id="conv2")

    # Check contexts
    context1 = query_engine._get_conversation_context("user1_conv1")
    context2 = query_engine._get_conversation_context("user2_conv2")

    assert len(context1) == 2
    assert context1[0]["query"] == "Query 1"
    assert context1[1]["query"] == "Query 2"

    assert len(context2) == 1
    assert context2[0]["query"] == "Query A"
