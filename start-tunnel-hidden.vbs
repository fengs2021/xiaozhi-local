' Silent launcher for Cloudflare tunnel (xiaozhi.fengs.fun public fallback)
' Keep ASCII-only: WSH decodes .vbs as ANSI(GBK) without a UTF-16 BOM.
Dim sh
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "."
sh.Run """%CLOUDFLARED%"" tunnel --no-autoupdate --protocol http2 run --token YOUR_TUNNEL_TOKEN", 0, False
