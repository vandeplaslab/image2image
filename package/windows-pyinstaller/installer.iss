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
; Viewer context menu for simple image extensions.
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.czi\shell\image2image.viewer"; ValueType: string; ValueName: ""; ValueData: "Open with image2image Viewer"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.czi\shell\image2image.viewer"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExeName},0"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.czi\shell\image2image.viewer"; ValueType: string; ValueName: "MultiSelectModel"; ValueData: "Single"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.czi\shell\image2image.viewer\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExeName}"" -t viewer -f ""%1"""

Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.tiff\shell\image2image.viewer"; ValueType: string; ValueName: ""; ValueData: "Open with image2image Viewer"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.tiff\shell\image2image.viewer"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExeName},0"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.tiff\shell\image2image.viewer"; ValueType: string; ValueName: "MultiSelectModel"; ValueData: "Single"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.tiff\shell\image2image.viewer\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExeName}"" -t viewer -f ""%1"""

Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.scn\shell\image2image.viewer"; ValueType: string; ValueName: ""; ValueData: "Open with image2image Viewer"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.scn\shell\image2image.viewer"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExeName},0"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.scn\shell\image2image.viewer"; ValueType: string; ValueName: "MultiSelectModel"; ValueData: "Single"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.scn\shell\image2image.viewer\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExeName}"" -t viewer -f ""%1"""

Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.qptiff\shell\image2image.viewer"; ValueType: string; ValueName: ""; ValueData: "Open with image2image Viewer"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.qptiff\shell\image2image.viewer"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExeName},0"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.qptiff\shell\image2image.viewer"; ValueType: string; ValueName: "MultiSelectModel"; ValueData: "Single"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.qptiff\shell\image2image.viewer\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExeName}"" -t viewer -f ""%1"""

Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.svs\shell\image2image.viewer"; ValueType: string; ValueName: ""; ValueData: "Open with image2image Viewer"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.svs\shell\image2image.viewer"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExeName},0"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.svs\shell\image2image.viewer"; ValueType: string; ValueName: "MultiSelectModel"; ValueData: "Single"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.svs\shell\image2image.viewer\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExeName}"" -t viewer -f ""%1"""

Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.ndpi\shell\image2image.viewer"; ValueType: string; ValueName: ""; ValueData: "Open with image2image Viewer"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.ndpi\shell\image2image.viewer"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExeName},0"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.ndpi\shell\image2image.viewer"; ValueType: string; ValueName: "MultiSelectModel"; ValueData: "Single"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.ndpi\shell\image2image.viewer\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExeName}"" -t viewer -f ""%1"""

; Viewer context menu for compound QPTIFF extensions.
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.raw\shell\image2image.viewer"; ValueType: string; ValueName: ""; ValueData: "Open with image2image Viewer"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.raw\shell\image2image.viewer"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExeName},0"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.raw\shell\image2image.viewer"; ValueType: string; ValueName: "MultiSelectModel"; ValueData: "Single"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.raw\shell\image2image.viewer"; ValueType: string; ValueName: "AppliesTo"; ValueData: "System.FileName:~>"".qptiff.raw"""
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.raw\shell\image2image.viewer\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExeName}"" -t viewer -f ""%1"""

Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.intermediate\shell\image2image.viewer"; ValueType: string; ValueName: ""; ValueData: "Open with image2image Viewer"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.intermediate\shell\image2image.viewer"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExeName},0"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.intermediate\shell\image2image.viewer"; ValueType: string; ValueName: "MultiSelectModel"; ValueData: "Single"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.intermediate\shell\image2image.viewer"; ValueType: string; ValueName: "AppliesTo"; ValueData: "System.FileName:~>"".qptiff.intermediate"""
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.intermediate\shell\image2image.viewer\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExeName}"" -t viewer -f ""%1"""

; Elastix and Fiducials context menus for their compound JSON extensions.
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.json\shell\image2image.elastix"; ValueType: string; ValueName: ""; ValueData: "Open with image2image Elastix"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.json\shell\image2image.elastix"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExeName},0"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.json\shell\image2image.elastix"; ValueType: string; ValueName: "MultiSelectModel"; ValueData: "Single"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.json\shell\image2image.elastix"; ValueType: string; ValueName: "AppliesTo"; ValueData: "System.FileName:~>"".config.json"""
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.json\shell\image2image.elastix\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExeName}"" -t elastix -p ""%1"""

Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.json\shell\image2image.fiducials"; ValueType: string; ValueName: ""; ValueData: "Open with image2image Fiducials"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.json\shell\image2image.fiducials"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExeName},0"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.json\shell\image2image.fiducials"; ValueType: string; ValueName: "MultiSelectModel"; ValueData: "Single"
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.json\shell\image2image.fiducials"; ValueType: string; ValueName: "AppliesTo"; ValueData: "System.FileName:~>"".i2r.json"""
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.json\shell\image2image.fiducials\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExeName}"" -t register -p ""%1"""

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
