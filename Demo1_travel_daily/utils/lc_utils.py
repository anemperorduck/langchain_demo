import json

def inspect_message(msg):
    print(f"类型: {type(msg).__name__}")
    print(f"content: {repr(msg.content)}")
    
    if msg.tool_calls:
        print(f"\ntool_calls 数量: {len(msg.tool_calls)}")
        for i, tc in enumerate(msg.tool_calls):
            print(f"  [{i}] name: {tc['name']}")
            print(f"  [{i}] args: {json.dumps(tc['args'], indent=4, ensure_ascii=False)}")
    
    if msg.additional_kwargs:
        print(f"\nadditional_kwargs keys: {list(msg.additional_kwargs.keys())}")
