import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import openai
from knowledge.ingest import ingest_text_document
from api import UPLOAD_JOBS
from core.rate_limit_retry import run_with_rate_limit_retry

# Mock Graphiti client
class MockGraphiti:
    def __init__(self):
        self.driver = MagicMock()
        self.driver.execute_query = AsyncMock(return_value=MagicMock(records=[]))
        self.add_episode = AsyncMock()

@pytest.mark.asyncio
async def test_retry_wrapper_logic():
    """Test the retry wrapper in isolation."""
    mock_op = AsyncMock()
    error = openai.RateLimitError(
        message="Please try again in 0.1s.", 
        response=MagicMock(), 
        body=None
    )
    mock_op.side_effect = [error, error, "success"]
    
    callback = MagicMock()
    
    result = await run_with_rate_limit_retry(
        lambda: mock_op(),
        op_name="test_op",
        max_attempts=5,
        base_sleep=0.1,
        on_rate_limit=callback
    )
    
    assert result == "success"
    assert mock_op.call_count == 3
    assert callback.call_count == 2
    # Check if callback received correct sleep time (approx 0.1 + 0.5 = 0.6s from message parsing)
    args, _ = callback.call_args
    assert args[0] >= 0.6 


@pytest.mark.asyncio
async def test_ingest_flow_with_retry():
    """Test the full ingest flow with mocked Graphiti and RateLimitError."""
    graphiti = MockGraphiti()
    
    # 429 Error with specific retry-after
    error_429 = openai.RateLimitError(
        message="Rate limit reached. Please try again in 0.1s.", 
        response=MagicMock(), 
        body=None
    )
    
    # Fail 2 times, then succeed
    graphiti.add_episode.side_effect = [error_429, error_429, {"uuid": "123", "name": "Success"}]

    job_id = "test_job_retry"
    UPLOAD_JOBS[job_id] = {
        "status": "pending", 
        "stage": "starting",
        "timing": {}
    }

    # We patch update_upload_job in api module because ingest.py imports it from there
    with patch('api.update_upload_job') as mock_update:
        # We need to simulate the real update_upload_job behavior of updating UPLOAD_JOBS
        # so that the code doesn't break if it reads back, though ingest mostly writes.
        # But ingest_text_document calls it.
        
        def side_effect_update(jid, **kwargs):
            if jid in UPLOAD_JOBS:
                UPLOAD_JOBS[jid].update(kwargs)
        
        mock_update.side_effect = side_effect_update

        # Execute
        result = await ingest_text_document(
            graphiti, 
            "Test content for retry", 
            job_id=job_id,
            source_description="retry_test"
        )
        
        # Verify success
        assert result["status"] == "ok"
        assert graphiti.add_episode.call_count == 3
        
        # Verify status updates were called with 'rate_limited'
        # We look for calls where stage='rate_limited'
        rate_limit_calls = [
            call for call in mock_update.mock_calls 
            if 'stage' in call.kwargs and call.kwargs['stage'] == 'rate_limited'
        ]
        assert len(rate_limit_calls) == 2
        assert rate_limit_calls[0].kwargs['attempt'] == 1
        assert rate_limit_calls[1].kwargs['attempt'] == 2

