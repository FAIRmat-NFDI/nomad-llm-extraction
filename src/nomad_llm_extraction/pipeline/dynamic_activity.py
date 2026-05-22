# dynamic_worker.py
from collections.abc import Sequence

from temporalio import activity
from temporalio.common import RawValue


@activity.defn(dynamic=True)
async def universal_activity_router(args: Sequence[RawValue]) -> Any:
    # Extract meta-information about what the workflow tried to call
    activity_info = activity.info()
    target_function_name = activity_info.activity_type

    # Find the function in your database or local dictionary
    if target_function_name not in DYNAMIC_REGISTRY:
        raise ValueError(f'Function {target_function_name} is not registered.')

    func = DYNAMIC_REGISTRY[target_function_name]

    # Deserialize the payload inputs
    # Temporal passes raw payloads to dynamic handlers; we convert them back
    payload_converter = activity.payload_converter()
    resolved_args = [payload_converter.from_payload(arg.payload, Any) for arg in args]

    # Execute the raw function
    if asyncio.iscoroutinefunction(func):
        return await func(*resolved_args)
    return func(*resolved_args)
