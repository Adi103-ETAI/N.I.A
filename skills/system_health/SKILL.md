---
name: System Health Check
description: Check CPU, RAM, and Disk usage on the current system
platforms: ["windows", "linux"]
---

To check system health, use the appropriate command based on the operating system:

**Windows:**
1. CPU Usage: `wmic cpu get loadpercentage`
2. RAM Usage: `wmic os get FreePhysicalMemory,TotalVisibleMemorySize`
3. Disk Usage: `wmic logicaldisk get size,freespace,caption`

Alternative (PowerShell):
- CPU: `Get-Counter '\Processor(_Total)\% Processor Time'`
- RAM: `Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 10`

**Linux:**
1. CPU Usage: `top -b -n 1 | head -5` or `mpstat`
2. RAM Usage: `free -h`
3. Disk Usage: `df -h`

**Combined Command:**
- Windows: `systeminfo | findstr /C:"Total Physical Memory" /C:"Available Physical Memory"`
- Linux: `htop` (interactive) or `vmstat 1 5` (5-second sample)

Always report the results in a clear, readable format like:
- CPU: 45% usage
- RAM: 8.2 GB / 16 GB (51%)
- Disk (C:): 120 GB free / 500 GB total
