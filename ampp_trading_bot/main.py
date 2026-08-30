import asyncio
import logging
import sys
from core.ampp_aggregator import MarketDataAggregator
from core.ampp_agent import AMPPOrchestrator, SystemHalt

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)

logger = logging.getLogger("ampp.main")

async def main() -> None:
    """The central nervous system of the AMPP architecture."""
    logger.info("Initializing AMPP Trading Bot...")
    
    # The ThreadSafe/Async communication boundary between Layer 1 and Layer 2
    trigger_queue = asyncio.Queue(maxsize=10)
    
    # Initialize the separated layers
    aggregator = MarketDataAggregator(trigger_queue)
    orchestrator = AMPPOrchestrator(trigger_queue)
    
    logger.info("Starting Layer 1 (Micro-Aggregator) and Layer 2 (MCP Orchestrator) concurrently.")
    
    aggregator_task = asyncio.create_task(aggregator.start(), name="layer1-aggregator")
    orchestrator_task = asyncio.create_task(orchestrator.run(), name="layer2-orchestrator")

    try:
        # wait for whichever finishes/raises first — under normal operation
        # neither should ever return, so this always means one of them died
        done, pending = await asyncio.wait(
            {aggregator_task, orchestrator_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )

        # actually tear down whichever task is still running before doing
        # anything else
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        # now that both are confirmed stopped, re-raise whatever actually
        # went wrong so the outer except blocks below can handle it
        for task in done:
            task.result()

    except SystemHalt as halt_exc:
        logger.critical(
            "FATAL SYSTEM SHUTDOWN: MAX_CUMULATIVE_LOSS_PCT breached! "
            "Layer 1 has been cancelled and confirmed stopped. %s",
            halt_exc,
        )
        sys.exit(1)
    except Exception as exc:
        logger.error("An unhandled exception collapsed the system:", exc_info=True)
        sys.exit(2)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt. Shutting down gracefully.")
