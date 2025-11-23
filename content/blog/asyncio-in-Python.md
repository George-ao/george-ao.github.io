---
title: "Asyncio in Python"
date: 2025-11-22
---

Asyncio is useful when the tasks are **I/O-bound**, which means the bottleneck is not the CPU execution speed, but the waiting time for I/O operations (like network requests). For instance, if you want to send 1 million requests via OpenAI API, instead of using a for loop and waiting for each separate request, you can use asyncio to run many requests concurrently.

It is still a single process, single thread program, which means it cannot speed up CPU-bound tasks.

### Concept
- **event loop**: A brain, managing many tasks and deciding one task at each time to be executed. It decides which task to be executed and yields control to that task. It cannot **take** control back but waits until the task is finished or the task explicitly gives back the control by `await`.

- **coroutine**: 
Coroutine refers to a coroutine function or coroutine object. A coroutine function is defined with `async def`. Calling it returns a coroutine object instead of running the code. It does not run immediately.

- **task**: When you wrap a coroutine into a task (via `create_task` or `gather`), it is submitted to a loop to run concurrently in the background.

### Core Mechanism

`create_task` wraps a coroutine into a Task and submit it to the event loop. It returns immediately and does not run the code.

`await` yields control back to the event loop. If you `await` a Task, the loop can run other scheduled tasks. If you `await` a coroutine directly, it runs immediately and then schedule other submitted tasks.

### Example
```python 
import asyncio

async def old_task():
    print("Old Task: I was scheduled first! Let me run!")

async def new_coro():
    print("New Coro: I am running directly! I cut in line!")

async def main1():
    task = asyncio.create_task(old_task())
    print("Main: Task scheduled.")
    await new_coro()

async def main2():
    task = asyncio.create_task(old_task())
    task2 = asyncio.create_task(new_coro())
    await task2

asyncio.run(main2())

# main1 result
New Coro: I am running directly! I cut in line!
Old Task: I was scheduled first! Let me run!
# main2 result
Old Task: I was scheduled first! Let me run!
New Coro: I am running directly! I cut in line!
```


### reference
<a href="https://docs.python.org/3/howto/a-conceptual-overview-of-asyncio.html#a-conceptual-overview-of-asyncio" target="_blank">A Conceptual Overview of asyncio</a> 