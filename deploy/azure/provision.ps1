<#
.SYNOPSIS
    Provision ONE small Azure VM for the Docker Compose stack. Dry run by default.

.DESCRIPTION
    Without -Execute this script only PRINTS the Azure CLI commands, so nothing is
    created and no credit is spent. Review the commands, check prices for your region
    in the Azure pricing calculator, then re-run with -Execute when you decide to deploy.

.EXAMPLE
    ./deploy/azure/provision.ps1 -DnsLabel brainmri-demo-1234 -AllowedSshIp 203.0.113.7
    ./deploy/azure/provision.ps1 -DnsLabel brainmri-demo-1234 -AllowedSshIp 203.0.113.7 -Execute
#>
param(
    [Parameter(Mandatory)] [string] $DnsLabel,          # becomes <label>.<region>.cloudapp.azure.com
    [Parameter(Mandatory)] [string] $AllowedSshIp,      # your public IP; SSH is closed to everyone else
    [string] $ResourceGroup = "rg-brain-mri",
    [string] $Location = "southafricanorth",
    [string] $VmName = "vm-brain-mri",
    [string] $VmSize = "Standard_B2as_v2",              # 2 vCPU / 8 GiB; matches the deployed student VM
    [string] $AutoShutdownUtc = "1900",                 # daily automatic shutdown (HHmm, UTC)
    [switch] $Execute
)

$ErrorActionPreference = "Stop"
$cloudInit = Join-Path $PSScriptRoot "cloud-init.yaml"

$commands = @(
    "az group create --name $ResourceGroup --location $Location",
    "az vm create --resource-group $ResourceGroup --name $VmName --image Ubuntu2404 --size $VmSize --admin-username azureuser --generate-ssh-keys --public-ip-sku Standard --public-ip-address-dns-name $DnsLabel --os-disk-size-gb 64 --custom-data `"$cloudInit`" --nsg-rule NONE",
    "az network nsg rule create --resource-group $ResourceGroup --nsg-name ${VmName}NSG --name allow-https --priority 100 --access Allow --protocol Tcp --destination-port-ranges 80 443",
    "az network nsg rule create --resource-group $ResourceGroup --nsg-name ${VmName}NSG --name allow-ssh-from-me --priority 110 --access Allow --protocol Tcp --destination-port-ranges 22 --source-address-prefixes $AllowedSshIp/32",
    "az vm auto-shutdown --resource-group $ResourceGroup --name $VmName --time $AutoShutdownUtc"
)

if (-not $Execute) {
    Write-Host "DRY RUN - nothing will be created. Commands that -Execute would run:`n" -ForegroundColor Yellow
    $commands | ForEach-Object { Write-Host "  $_" }
    Write-Host "`nAfter creation: ssh azureuser@$DnsLabel.$Location.cloudapp.azure.com and follow deploy/azure/README.md." -ForegroundColor Yellow
    return
}

foreach ($command in $commands) {
    Write-Host "> $command" -ForegroundColor Cyan
    Invoke-Expression $command
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $command" }
}
Write-Host "VM ready at https://$DnsLabel.$Location.cloudapp.azure.com once the stack is started (see README)." -ForegroundColor Green
