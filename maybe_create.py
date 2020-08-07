# Interacts with conda's env creating to only create
# an env if it doesn't already exist.
# This is a super overkill. Can be reused however.
import sys

import click
import pexpect

# Run the bash command 
child = pexpect.spawn('conda create --name eli5')

# Apply different regex patterns to the response to know in which 
# of the possible cases we are in.
i = child.expect(['Remove existing environment.*', 'Proceed.*'])

# Print the text before what we matched
print('Before:', child.before)

# Send 'y' if it prints ~'Proceed with creation?' 
# Send 'n' if it prints ~'Remove existing env?'
if i == 0:
    sending = 'n'
    color = 'red'
elif i == 1:
    sending = 'y'
    color = 'green'
else:
    raise

# Print the text that we matched
print(click.style(f'Matched: {child.after}', fg=color))

# Print what we are sending
print(click.style(f'Sending: {sending}', fg=color))
# Send the response
child.sendline(sending)

# Wait for the bash command to quit
child.expect(pexpect.EOF)
child.close()