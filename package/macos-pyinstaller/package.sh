#!/bin/sh

# sign app with hash of my securityDeveloper ID Application
codesign \
  --deep \
  --force \
  --options=runtime \
  --entitlements ./entitlements.plist \
  --sign "8851DF26E43B9A5C81AE4E2FB087EDE22AADE914" \
  --timestamp \
  ./dist/image2image.app

# remove existing file
rm -r dist/dmg/*
test -f "dist/image2image.dmg" && rm "dist/image2image.dmg"

# create dmg file that user can drag the App.app to Applications using hdiutil
ln -s /Applications dist/dmg/Applications
hdiutil create \
  -volname "image2image" \
  -srcfolder ./dist/dmg \
  -ov \
  -format UDZO \
  ./dist/image2image.dmg



# package as pkg for installation using ditto
#ditto \
#  --noqtn \
#  --rsrc \
#  ./dist/image2image.app \
#  ./dist/tmp

#ditto \
#  -v \
#  ./dist/image2image.app \
#  ./dist/tmp
#
## remove existing file
#test -f "dist/image2image.pkg" && rm "dist/image2image.pkg"
#
## build the package with hash of my securityDeveloper ID Installer
#productbuild \
#  --identifier "com.vandeplaslab.image2image" \
#  --sign "CC24F4E725EF39EEFC64D0E479580752CA55E46E" \
#  --timestamp \
#  --root ./dist/tmp /Applications ./dist/image2image.pkg

#productbuild --identifier "com.vandeplaslab.image2image" --sign "CC24F4E725EF39EEFC64D0E479580752CA55E46E" --timestamp --root ./dist/image2image/image2image.pkg ./dist/image2image.pkg
#productbuild --identifier "com.vandeplaslab.image2image" --sign "CC24F4E725EF39EEFC64D0E479580752CA55E46E" --timestamp --root /tmp/myapp / image2image.pkg

# notarize the package
xcrun \
  notarytool \
  submit \
  --verbose \
  --wait \
  --keychain-profile "macos-notary" \
  ./dist/image2image.dmg