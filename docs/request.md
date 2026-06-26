# Feature Requests / Bug Reports

You can request new features and report bugs in a couple of different ways.

1. Send a request using the `Help > Send feedback or request...` menu option. This will open a popup window where you can provide more information. We are using [sentry.io](https://sentry.io/welcome/) under the hood.
2. Create a GitHub issue on the [image2image-docs](https://github.com/vandeplaslab/image2image-docs) public website. (Requires a GitHub account)
3. Contact Lukasz G. Migas using Slack or email.


## Bug monitoring

The app uses [sentry.io](https://sentry.io/welcome/) for automatic bug reports. This sends a HTML request with error tracebacks to sentry so that we can figure out what went wrong. You can disable this behaviour by using the `Help > Telemetry...` menu option and then selecting `No, I'd prefer not to send bug reports.`.

!!! note Note
        While we try not to capture any personal information, this is not guaranteed and some information such as file paths might be sent to us. We don't use this information for anything other than figuring out where we have have bugs and what causes the errors to happen.