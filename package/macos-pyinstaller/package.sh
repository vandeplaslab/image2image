#!/bin/sh

# sign app with hash of my securityDeveloper ID Application
echo "Signing app..."
codesign \
  --deep \
  --force \
  --options=runtime \
  --entitlements ./entitlements.plist \
  --sign "8851DF26E43B9A5C81AE4E2FB087EDE22AADE914" \
  --timestamp \
  ./dist/image2image.app
echo "Signed app"

# remove existing file
mkdir -p dist/dmg
rm -r dist/dmg/*

# create dmg file that user can drag the App.app to Applications using hdiutil
# create a link to Applications folder so they can easily drag the app to Applications
ln -s /Applications dist/dmg/Applications

# copy ap to dmg folder
mv "dist/image2image.app" dist/dmg

# create dmg file
echo "Creating dmg..."
hdiutil create \
  -volname "image2image" \
  -srcfolder ./dist/dmg \
  -ov \
  -format UDZO \
  ./dist/image2image.dmg
echo "Created dmg"

# notarize the package
# can also add --wait \ to wait for notarization to complete
#xcrun \
#  notarytool \
#  submit \
#  --verbose \
#  --keychain-profile "macos-notary" \
#  ./dist/image2image.dmg
#echo "Notarized app"
#
## staple the ticket
#xcrun \
#  stapler \
#  staple \
#  ./dist/image2image.dmg
#echo "Stapled ticket"

# rename
python rename.py
echo "Renamed app"