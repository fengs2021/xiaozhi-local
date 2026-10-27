@echo off
chcp 65001 >nul
"%CLOUDFLARED%" tunnel --no-autoupdate --protocol http2 run --token YOUR_TUNNEL_TOKEN >> "cf-tunnel.log" 2>&1
