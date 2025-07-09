---
title: Set up Sunshine steaming in Windows Hyper-V with GPU partitioning
tags: 'hyperv, sunshine, gaming'
categories:
  - Fun
date: 2025-04-10 15:48:18
---

# Set up Sunshine steaming in Windows Hyper-V with GPU Parititoning enabled

Why gaming on a virtual machine?

Gaming on a virtual machine (VM) can be an option for several reasons:
- **Isolation**: Running games in a VM can provide a layer of isolation from the host system, which can be useful for security and stability.
- **Flexibility**: VMs can be easily created, modified, and deleted, allowing for quick experimentation with different configurations or setups.

<!-- more -->
## Pre-requisites
- A Windows machine with Hyper-V enabled

I will briefly go over the steps to set up a Windows 10 VM.
1. Make sure you have Hyper-V(Not available on Windows 10/11 Home) enabled through Control Panel 
2. Getting an ISO image of Windows 10/11, either through the Microsoft website or through other means.
3. Open Hyper-V Manager and create a new VM, assigning reasonable resources to it.
4. Attach the Windows 10/11 ISO image to the VM and start it (Tips: you might need to disable Secure Boot in the VM settings and try tapping keys when you first boot the image).
5. Proceed with the installation of Windows 10/11.

## GPU Partitioning adopted from [This guide](https://www.youtube.com/watch?v=KDc8lbE2I6I)
The GPU can be shared with the host and the VM, alternatively, you can try to enable GPU passthrough, but it is not covered in this guide.

My Machine has 2 GPUs, one is the integrated AMD GPU and the other is a dedicated NVIDIA GPU. I will need to disable the integrated GPU in the Device Manager and assign the NVIDIA GPU to the VM.

1. Open Device Manager and disable the integrated GPU.
2. Mount the VM disk file by simply opening the file in Windows Explorer and assume it is the new `F:` drive.
3. Create a folder called `HostDriverStore` in `F:\Windows\System32\` (This is the mounted drive).
3. Copy the whole folder `C:/Windows/System32/DriverStore/FileRepository` to `F:\Windows\System32\HostDriverStore`.
4. (Nvidia only) Copy everyfile starts with `nv` in `C:\Windows\System32\` to `F:\Windows\System32\`.
5. (AMD) Amd probably has a similar folder, but I don't have an AMD GPU to test it.

Next we will need to paritition and assign the GPU to the VM. 
1. Open PowerShell ISE as administrator.
2. Change the name of the `vm` variable to the name of your VM in the script below.
3. Change the RAM size to your liking.
3. Run the script below 

```powershell
$vm = "Game"
if (Get-VMGpuPartitionAdapter -VMName $vm -ErrorAction SilentlyContinue) {
   Remove-VMGpuPartitionAdapter -VMName $vm
}

Set-VM -GuestControlledCacheTypes $true -VMName $vm
Set-VM -LowMemoryMappedIoSpace 1Gb -VMName $vm
Set-VM -HighMemoryMappedIoSpace 32Gb -VMName $vm
Add-VMGpuPartitionAdapter -VMName $vm
```

Last step before we start the machine is to disable the `Checkpoint` feature in Hyper-V.
1. Open Hyper-V Manager and select the VM.
2. Click on `Settings` and disable `Checkpoints`.
3. Now start the VM and you should see the graphic card in in Device Manager.


# Install Virtual Audio Device
We need a virtual audio device for sunshine to steam audio. You can use the Virtual Audio Device from the `VB-Cable` project or the `Virtual Audio Cable` project.
1. Download either `VB-Cable` or `Virtual Audio Cable` and Install it.
2. You should see the virtual audio device in the device manager (it will not appear in `Sound` settings)

# LAN access 
If you want to access the VM from another device in local network, you can do port forwarding (not covered here) or simply add an external network adapter to the VM.
1. Open Hyper-V Manager and select the Network Switch on the right side
2. Click on `Virtual Switch Manager` and create a new `External` network switch.
3. Toggle the share network adapter option
4. Open the VM settings and add a new network adapter and select the external network switch.

# Install Sunshine
Last step is to install sunshine on the VM and you should be able to connect to it.





