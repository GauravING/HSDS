# PowerShell quick health and connectivity checks for the Helmet/Seatbelt project
# Usage: Open PowerShell, cd to backend folder and run: .\tools\check_stack.ps1

$base = 'http://127.0.0.1:8000'
Write-Host "Base URL: $base`n"

function Try-Invoke([string]$method, [string]$path, $body=$null, $headers=@{}){
    $uri = "$base$path"
    Write-Host "--> $method $uri"
    try{
        if ($body -ne $null) {
            $json = $body | ConvertTo-Json -Depth 5
            $resp = Invoke-RestMethod -Uri $uri -Method $method -Body $json -Headers $headers -ContentType 'application/json' -ErrorAction Stop
            Write-Host "Response (parsed):"
            $resp | ConvertTo-Json -Depth 6 | Write-Host
            return $resp
        } else {
            $resp = Invoke-RestMethod -Uri $uri -Method $method -Headers $headers -ErrorAction Stop
            Write-Host "Response (parsed):"
            $resp | ConvertTo-Json -Depth 6 | Write-Host
            return $resp
        }
    } catch {
        Write-Host "ERROR: $_.Exception.Message" -ForegroundColor Red
        if ($_.Exception.Response) {
            try{
                $r = $_.Exception.Response.GetResponseStream()
                $sr = New-Object System.IO.StreamReader($r)
                $txt = $sr.ReadToEnd()
                Write-Host "Raw response body:" -ForegroundColor Yellow
                Write-Host $txt
            } catch {}
        }
        return $null
    }
}

# 1) Health check (app + DB)
Try-Invoke -method GET -path '/api/health'

# 2) Allowed emails
Try-Invoke -method GET -path '/auth/allowed_emails'

# 3) Signup GET informational
Try-Invoke -method GET -path '/auth/signup'

# 4) If you want to test signup + login + profile, uncomment and edit the email/password.
#$testEmail = 'devuser@example.com'
#Test signup (POST)
#$payload = @{ username = 'devuser'; email = $testEmail; password = 'DevPass123'; full_name = 'Dev User' }
#Try-Invoke -method POST -path '/auth/signup' -body $payload

# Test login (POST) - adjust email/password if you created a user
#$login = @{ email = $testEmail; password = 'DevPass123' }
#$loginResp = Try-Invoke -method POST -path '/auth/login' -body $login

# If login successful, call profile with Authorization header
# if ($loginResp -and $loginResp.access_token) {
#    $token = $loginResp.access_token
#    $hdr = @{ Authorization = "Bearer $token" }
#    Try-Invoke -method GET -path '/auth/profile' -headers $hdr
# }

# 5) Optional: test detect debug upload using a small image file (adjust path)
# $imagePath = 'C:\path\to\sample.jpg'
# if (Test-Path $imagePath) {
#    Write-Host "Uploading $imagePath to /detect/debug"
#    try{
#        $uri = "$base/detect/debug"
#        $form = @{ file = Get-Item $imagePath }
#        $r = Invoke-RestMethod -Uri $uri -Method Post -Form $form -ErrorAction Stop
#        $r | ConvertTo-Json -Depth 6 | Write-Host
#    } catch {
#        Write-Host "Upload error: $_" -ForegroundColor Red
#    }
# } else { Write-Host "(Skip detect debug: update `\$imagePath` in this script and uncomment to test)" }

Write-Host "\nDone. Review outputs above. If anything returns an error, copy the text and paste it to me and I'll help interpret it."