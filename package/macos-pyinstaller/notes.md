
# Validate notarization credentials

xcrun notarytool store-credentials macos-notary --apple-id lukas.migas@yahoo.com --team-id ZC898C78QL --password <GET FROM KEYCHAIN>

# To use notarization, we will need to use "macos-notary" keychain

--keychain-profile "macos-notary
