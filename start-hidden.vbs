' Silent launcher for XiaoZhi local control center (window fully hidden)
' NOTE: keep this file ASCII-only. WSH reads .vbs as ANSI(GBK) unless it has a UTF-16 BOM,
' so UTF-8 Chinese comments get mis-decoded and can swallow the next line.
Dim sh
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "."
sh.Run """%PYTHON%"" run.py", 0, False
