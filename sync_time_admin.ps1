# Run this script as Administrator to configure 15-minute NTP synchronization.
Write-Host "Configuring Windows Time Service (w32time) to synchronize every 15 minutes..." -ForegroundColor Cyan

# Set Google NTP server and others
w32tm /config /manualpeerlist:"time.google.com,0x9 time.windows.com,0x9" /syncfromflags:manual /update

# Change polling interval to 900 seconds (15 minutes) in registry
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\W32Time\TimeProviders\NtpClient" -Name "SpecialPollInterval" -Value 900

# Restart w32time service to apply changes
Restart-Service w32time

# Force an immediate resync
w32tm /resync /force

Write-Host "Windows Time Service has been configured successfully!" -ForegroundColor Green
