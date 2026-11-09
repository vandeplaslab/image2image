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
#define IconFile "..\..\src\image2image\assets\icon.ico"
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
ChangesAssociations=yes
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
Name: "{group}\{#AppName} Viewer"; Filename: "{app}\{#AppExeName}"; Parameters: "-t viewer"; WorkingDir: "{app}"
Name: "{group}\{#AppName} Elastix"; Filename: "{app}\{#AppExeName}"; Parameters: "-t elastix"; WorkingDir: "{app}"
Name: "{group}\{#AppName} Convert"; Filename: "{app}\{#AppExeName}"; Parameters: "-t convert"; WorkingDir: "{app}"
Name: "{group}\{#AppName} Fiducials"; Filename: "{app}\{#AppExeName}"; Parameters: "-t register"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Registry]
; Emit each context menu from one shared template. Compound extensions keep
; their filename filter so unrelated .raw, .intermediate, and .json files are unaffected.
#define Extension ".czi"
#define Tool "viewer"
#define Label "Viewer"
#define FileOption "f"
#define Filter ""
#define ContextKey() "Software\Classes\SystemFileAssociations\" + Extension + "\shell\image2image." + (Tool == "register" ? "fiducials" : Tool)

#sub ContextMenu
Root: HKA; Subkey: "{#ContextKey()}"; ValueType: string; ValueName: ""; ValueData: "Open with {#AppName} {#Label}"; Flags: uninsdeletekey
Root: HKA; Subkey: "{#ContextKey()}"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExeName},0"
Root: HKA; Subkey: "{#ContextKey()}"; ValueType: string; ValueName: "MultiSelectModel"; ValueData: "Single"
#if Filter != ""
Root: HKA; Subkey: "{#ContextKey()}"; ValueType: string; ValueName: "AppliesTo"; ValueData: "System.FileName:~>""{#Filter}"""
#endif
Root: HKA; Subkey: "{#ContextKey()}\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExeName}"" -t {#Tool} -{#FileOption} ""%1"""
#endsub

#emit ContextMenu
#expr Extension = ".tiff"
#emit ContextMenu
#expr Extension = ".scn"
#emit ContextMenu
#expr Extension = ".qptiff"
#emit ContextMenu
#expr Extension = ".ndpi"
#emit ContextMenu
#expr Extension = ".svs"
#emit ContextMenu
#expr Extension = ".raw", Filter = ".qptiff.raw"
#emit ContextMenu
#expr Extension = ".intermediate", Filter = ".qptiff.intermediate"
#emit ContextMenu
#expr Extension = ".json", Tool = "elastix", Label = "Elastix", FileOption = "p", Filter = ".config.json"
#emit ContextMenu
#expr Tool = "register", Label = "Fiducials", Filter = ".i2r.json"
#emit ContextMenu

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
