grantPermissions:
	chmod 766 ../bin/*

run:
	../bin/run.sh

runCelery:
	celery -A lib.celery.App worker -l info

runA:
	../bin/run.sh -a

# This is a quick check for timming.
# For thorough results, use the profile
# option
runT:
	echo "Simple timing analysis"
	../bin/run.sh -t

clean:
	../bin/clean.sh

# ---- Easy way of checking the functions ---------
checkLog:
	../bin/checkLog.sh

checkTime:
	../bin/checkTime.sh

checkCeleryLog:
	../bin/checkCeleryLog.sh

checkCeleryTime:
	../bin/checkCeleryTime.sh

test:
	python3 -m pytest ../tests

define string_to_insert
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd:
	html_theme = 'default'
else:  # only import and set the theme if we're building docs locally
	import sphinx_rtd_theme
	html_theme = 'sphinx_rtd_theme'
	html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
endef
export string_to_insert

doc:
	@echo 'Generating a fresh copy of the docs ...'
	@rm -rf ../docs
	@mkdir ../docs
	@sphinx-apidoc -F -o ../docs .
	@echo "autoclass_content = 'both'" >> ../docs/conf.py
	@sed "s/^extensions.*/& 'sphinx.ext.napoleon','celery.contrib.sphinx',/"        ../docs/conf.py   > ../docs/conf2.py
	@sed "s/^# import/import/"                              ../docs/conf2.py  > ../docs/conf3.py
	@sed "s/^# sys.path./sys.path./"                        ../docs/conf3.py  > ../docs/conf4.py
	@sed "s|^html_theme =|#html_theme =|"  					../docs/conf4.py  > ../docs/conf5.py
	@echo '# napoleon extensions over here '                >> ../docs/conf1.py
	@echo '# ------------------------------'                >> ../docs/conf1.py
	@echo 'napoleon_google_docstring = True'                >> ../docs/conf1.py
	@echo 'napoleon_numpy_docstring = True'                 >> ../docs/conf1.py
	@echo 'napoleon_include_init_with_doc = False'          >> ../docs/conf1.py
	@echo 'napoleon_include_private_with_doc = False'       >> ../docs/conf1.py
	@echo 'napoleon_include_special_with_doc = True'        >> ../docs/conf1.py
	@echo 'napoleon_use_admonition_for_examples = False'    >> ../docs/conf1.py
	@echo 'napoleon_use_admonition_for_notes = False'       >> ../docs/conf1.py
	@echo 'napoleon_use_admonition_for_references = False'  >> ../docs/conf1.py
	@echo 'napoleon_use_ivar = False'                       >> ../docs/conf1.py
	@echo 'napoleon_use_param = True'                       >> ../docs/conf1.py
	@echo 'napoleon_use_rtype = True'                       >> ../docs/conf1.py
	@echo ''                                                >> ../docs/conf1.py
	@echo ''                                                >> ../docs/conf1.py
	@echo "$$string_to_insert" 								>> ../docs/conf1.py
	@cat ../docs/conf5.py ../docs/conf1.py > ../docs/conf6.py
	@rm ../docs/conf.py ../docs/conf1.py ../docs/conf2.py ../docs/conf3.py ../docs/conf4.py ../docs/conf5.py
	@mv ../docs/conf6.py ../docs/conf.py
	@echo 'Generating the doc tree ....'
	@echo '----------------------------'
	@sphinx-build -b html -aE -d ../docs/doctrees -c ../docs ../docs ../docs/_build/html
	@cp -R ../docs/_build/html ../docs1
	@rm -rf ../docs
	@mv ../docs1 ../docs
	@touch ../docs/.nojekyll

