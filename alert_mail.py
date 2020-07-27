from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import geocoder

def do_this(charles,bin_laden):
	strFrom = '073bct525.paridhi@pcampus.edu.np'
	strTo = 'parajuli.paridhi@gmail.com'
	g = geocoder.ip('me')

	msgRoot = MIMEMultipart('related')
	msgRoot['Subject'] = 'Alert! Criminal Suspected.'
	msgRoot['From'] = 'your email'
	msgRoot['To'] = 'reciever email'
	#msgRoot.preamble = 'This is a multi-part message in MIME format.'

	msgAlternative = MIMEMultipart('alternative')
	msgRoot.attach(msgAlternative)

	msgBody = MIMEText('Criminal(s) has been suspected. Please find the attached photo(s) of the suspected. The lattitude and longitude of the suspected are: ')
	msgLocation=MIMEText('location:'+str(g.latlng), 'plain')
	msgAlternative.attach(msgBody)
	

	msgText = MIMEText('<b>Criminal(s) has been suspected. Please find the attached photo(s) of the suspected. The lattitude and longitude of the suspected are:</b><br><br>'
                   '<img src="cid:image1">'
                   '<br>'
                   '<br>'
                   '<img src="cid:image2">'
                   '<br>'
                   '<br>'
                   'Sending Attachment', 'html')
	
	msgAlternative.attach(msgText)
	msgAlternative.attach(msgLocation)
	if bin_laden==1:
		fp = open('bin_laden.jpg', 'rb')
		msgImage1 = MIMEImage(fp.read())
		fp.close()
		msgImage1.add_header('Content-ID', '<image1>')
		msgRoot.attach(msgImage1)
	if charles==1:
		fp = open('charles.jpg', 'rb')
		msgImage2 = MIMEImage(fp.read())
		fp.close()
		msgImage2.add_header('Content-ID', '<image2>')
		msgRoot.attach(msgImage2)

	import smtplib
	smtp = smtplib.SMTP('smtp.gmail.com', 587)
	smtp.starttls() 
	smtp.login(strFrom, "your password")
	smtp.sendmail(strFrom, strTo, msgRoot.as_string())
	print('sent')
	smtp.quit()

