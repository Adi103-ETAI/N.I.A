"""TARA Units - Auto-discovered tool modules.

Place tool modules here. Any function decorated with @tara_tool
will be automatically discovered by the ToolRegistry.

Example module (hardware.py):
    from tara.protocols import tara_tool
    
    @tara_tool(
        name="system_stats",
        category="system",
        description="Get CPU and RAM usage"
    )
    def system_stats() -> str:
        import psutil
        return f"CPU: {psutil.cpu_percent()}%, RAM: {psutil.virtual_memory().percent}%"
"""
