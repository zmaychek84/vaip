.. 
   Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
   
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

========
DocTitle
========

Then use underlining only:

..  _header1:


part 1
######

chapter 1
*********

Header 1 sections
=================

Header 1.1
----------

Header 1.1.1
~~~~~~~~~~~~

Header 1.1.1.1
""""""""""""""

link to head1 :ref:`header1`
:ref:`pattern-label`

inline code :code:`1 + 2`


The standard reST inline markup is quite simple: use

one asterisk: *text* for emphasis (italics),

two asterisks: **text** for strong emphasis (boldface), and

backquotes: ``text`` for code samples, variables, and literals.


* This is a bulleted list.
* It has two items, the second
  item uses two lines.

1. This is a numbered list.
2. It has two items too.

#. This is a numbered list.
#. It has two items too.

* this is
* a list

  * with a nested list
  * and some subitems

* and here the parent list continues


term (up to a line of text)
   Definition of the term, which must be indented

   and can even consist of multiple paragraphs

next term
   Description.


This is a normal text paragraph. The next paragraph is a code sample ::

   It is not processed in any way, except
   that the indentation is removed.

   It can span multiple lines.
   
   :: comment
   this is :: adf

This is a normal text paragraph again.


::

this `Google Search <http://google.com>`_

.. cpp:function:: foo(x)
              foo(y, z)
   
   Return a line of text input from the user.


Lorem ipsum [#]_ dolor sit amet ... [#]_

.. rubric:: Footnotes

.. [#] Text of the first footnote.
.. [#] Text of the second footnote.


The area of a circle is :math:`A_\text{c} = (\pi/4) d^2`.


.. seealso::

   Module :py:mod:`zipfile`
      Documentation of the :py:mod:`zipfile` standard module.

   `GNU tar manual, Basic Tar Format <http://link>`_
      Documentation for tar archive files, including GNU tar extensions.


.. hlist::
   :columns: 3

   * A list of
   * short items
   * that should be
   * displayed
   * horizontally


.. code-block:: c

    int main(int argc, char * argv[]) {
       printf("hello world");
       return 0;
    }

.. literalinclude:: ../../vaip/etc/vaip_config.json
    :linenos:
    :lines: 1,3,5-10
    :language: JSON
