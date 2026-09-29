import asyncio
from streamlit.web.cli import main

# Use the Windows Selector event loop instead of Proactor
asyncio.set_event_loop_policy(
    asyncio.WindowsSelectorEventLoopPolicy()
)

if __name__ == "__main__":
    main()