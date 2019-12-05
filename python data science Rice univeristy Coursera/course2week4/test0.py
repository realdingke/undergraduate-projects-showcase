def update_line(line):
    """
    Takes a string line representing a single line of code
    and returns a string with print updated
    """
    string1=line.lstrip();
    if "print" in string1:
        string2="print"+"("+string1[6:]+")";
        string3=line[0:line.index("p")]+string2;
        return string3
    else:
        return line

def update_pre_block(pre_block):
    """
    Take a string that correspond to a <pre> block in html and parses it into lines.  
    Returns string corresponding to updated <pre> block with each line
    updated via process_line()
    """
    updated_block="";
    count=0;
    for line in pre_block.splitlines():
        count+=1;
        if count<len(pre_block.splitlines()):
            line=update_line(line)+"\n";
            updated_block=updated_block+line;
        else:
            line=update_line(line);
            updated_block=updated_block+line;
    return updated_block

text="""               </a>
            </h4>
          </div>
          <div id='Integers-Floats-constants-collapse' class='panel-collapse collapse in' role='tabpanel' aria-labelledby='Integers-Floats-constants'>
            <div class='panel-body'>              <div>
                <dl class="dl-horizontal">
                  <dt>Examples:</dt><dd><table class='table table-condensed table-hover'><thead><tr><th>Code</th><th>Output</th></tr></thead><tbody><tr><td><pre class='cm'>print 12</pre></td><td><pre>12</pre></td></tr><tr><td><pre class='cm'>print -12.0</pre></td><td><pre>-12</pre></td></tr><tr><td><pre class='cm'>print 0b1001</pre></td><td><pre>9</pre></td></tr><tr><td><pre class='cm'>print 021</pre></td><td><pre>17</pre></td></tr><tr><td><pre class='cm'>print -021</pre></td><td><pre>-17</pre></td></tr><tr><td><pre class='cm'>print 0x3A8</pre></td><td><pre>936</pre></td></tr><tr><td><pre class='cm'>print 12L</pre></td><td><pre>12</pre></td></tr><tr><td><pre class='cm'>print 12345678901234567890123456789012345678901234567890</pre></td><td><pre>12345678901234567890123456789012345678901234567890</pre></td></tr><tr><td><pre class='cm'>print 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFL</pre></td><td><pre>340282366920938463463374607431768211455</pre></td></tr><tr><td><pre class='cm'>print +12.12345</pre></td><td><pre>12.12345</pre></td></tr><tr><td><pre class='cm'>print -12.12345</pre></td><td><pre>-12.12345</pre></td></tr><tr><td><pre class='cm'>print 1.2e3</pre></td><td><pre>1200</pre></td></tr><tr><td><pre class='cm'>print 1.2e-3</pre></td><td><pre>0.0012</pre></td></tr></tbody></table></dd>                </dl>
     """
text_updated="";
for line in text.split("</pre>"):
    if "pre" in line:
        line_pre=line.replace(line[0:line.rindex(">")+1],"");
        line_updated=update_pre_block(line_pre);
        line_total=line.replace(line[line.rindex(">")+1:],line_updated)+"</pre>";
        text_updated=text_updated+line_total;
    else:
        text_updated=text_updated+line;
print(text_updated)
