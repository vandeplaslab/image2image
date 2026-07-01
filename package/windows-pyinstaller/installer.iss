#ifndef AppName
#define AppName "image2image"
#endif

#ifndef AppExeName
#define AppExeName "image2image.exe"
#endif

#ifndef AppVersion
#define AppVersion "0.0.0"
#endif

#ifndef AppPublisher
#define AppPublisher "vandeplaslab"
#endif

#ifndef SourceDir
#define SourceDir "dist\image2image"
#endif

#ifndef OutputDir
#define OutputDir "dist"
#endif

#ifndef OutputBaseFilename
#define OutputBaseFilename "image2image-0.0.0-win_amd64-setup"
#endif

#ifndef IconFile
#define IconFile "src\image2image\assets\icon.ico"
#endif

[Setup]
AppId={{A5D1F8C2-47B9-4E6A-9F03-D2B7C1E8A64F}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL=https://github.com/vandeplaslab
AppSupportURL=https://github.com/vandeplaslab/image2image/issues
AppUpdatesURL=https://github.com/vandeplaslab/image2image
DefaultDirName={autopf}\image2image
DefaultGroupName={#AppName}
DisableProgramGroupPage=no
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseFilename}
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#AppExeName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=6

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
